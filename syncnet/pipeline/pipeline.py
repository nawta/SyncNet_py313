"""SyncNet pipeline with Python 3.13 optimizations."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Any
from collections.abc import Sequence

import cv2
import ffmpeg
import numpy as np
import torch
from scipy import signal
from scipy.interpolate import interp1d
from scenedetect import ContentDetector, SceneManager
from scenedetect.video_manager import VideoManager

from syncnet.core.models import SyncNetModel
from syncnet.core.inference import SyncNetInstance
from syncnet.core.types import Track, Detection, PipelineResult, SyncResult
try:
    from syncnet.detectors.s3fd.detector import S3FDNet
except ImportError:
    # Fall back to original implementation
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[2] / "script"))
    from detectors.s3fd.nets import S3FDNet
from syncnet.pipeline.config import PipelineConfig, AudioConfig, VideoConfig
from syncnet.utils.video import extract_audio, get_video_info
from syncnet.utils.face_detection import FaceTracker


logger = logging.getLogger(__name__)


class SyncNetPipeline:
    """Complete pipeline for audio-visual synchronization analysis.
    
    Args:
        config: Pipeline configuration
        device: Device to run models on
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        device: str = "cuda"
    ) -> None:
        self.config = config or PipelineConfig()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.face_detector = self._load_face_detector()
        self.syncnet = self._load_syncnet()
        
        # Initialize face tracker
        self.face_tracker = FaceTracker(
            min_track_length=self.config.min_track,
            min_face_size=self.config.min_face_size,
            max_failed_detections=self.config.num_failed_det
        )
    
    def _load_face_detector(self) -> S3FDNet:
        """Load S3FD face detector."""
        logger.info(f"Loading S3FD from {self.config.s3fd_weights}")
        
        detector = S3FDNet(device=str(self.device))
        if self.config.s3fd_weights.exists():
            # Load weights directly for compatibility with original implementation
            state_dict = torch.load(self.config.s3fd_weights, map_location=self.device)
            detector.load_state_dict(state_dict)
        else:
            logger.warning(f"S3FD weights not found at {self.config.s3fd_weights}")
        
        detector.eval()
        return detector
    
    def _load_syncnet(self) -> SyncNetInstance:
        """Load SyncNet model."""
        logger.info(f"Loading SyncNet from {self.config.syncnet_weights}")
        
        model = SyncNetModel()
        syncnet = SyncNetInstance(model, device=str(self.device))
        
        if self.config.syncnet_weights.exists():
            syncnet.load_weights(self.config.syncnet_weights)
        else:
            logger.warning(f"SyncNet weights not found at {self.config.syncnet_weights}")
        
        return syncnet
    
    async def process_video_async(
        self,
        video_path: Path | str,
        cache_dir: Optional[Path] = None,
        scene_detection: bool = True
    ) -> PipelineResult:
        """Process video asynchronously for synchronization analysis.
        
        Args:
            video_path: Path to input video
            cache_dir: Directory for temporary files
            scene_detection: Whether to perform scene detection
            
        Returns:
            Pipeline processing result
        """
        video_path = Path(video_path)
        
        # Create working directory
        if cache_dir:
            work_dir = cache_dir / f"syncnet_{video_path.stem}"
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            work_dir = Path(tempfile.mkdtemp(prefix="syncnet_"))
        
        try:
            # Extract audio asynchronously
            audio_task = asyncio.create_task(
                self._extract_audio_async(video_path, work_dir)
            )
            
            # Process video frames
            tracks = await self._process_video_frames_async(
                video_path, work_dir, scene_detection
            )
            
            # Wait for audio extraction
            audio_path = await audio_task
            
            # Process each track
            sync_results = []
            for track in tracks:
                result = await self._process_track_async(
                    track, audio_path, work_dir
                )
                sync_results.append(result)
            
            return PipelineResult(
                video_path=str(video_path),
                sync_results=sync_results,
                num_tracks=len(tracks),
                processing_time=0.0  # TODO: Add timing
            )
            
        finally:
            # Cleanup if using temporary directory
            if not cache_dir and work_dir.exists():
                shutil.rmtree(work_dir)
    
    def process_video(
        self,
        video_path: Path | str,
        cache_dir: Optional[Path] = None,
        scene_detection: bool = True
    ) -> PipelineResult:
        """Process video for synchronization analysis (synchronous version).
        
        Args:
            video_path: Path to input video
            cache_dir: Directory for temporary files
            scene_detection: Whether to perform scene detection
            
        Returns:
            Pipeline processing result
        """
        return asyncio.run(
            self.process_video_async(video_path, cache_dir, scene_detection)
        )
    
    async def _extract_audio_async(
        self,
        video_path: Path,
        work_dir: Path
    ) -> Path:
        """Extract audio from video asynchronously."""
        audio_path = work_dir / "audio.wav"
        
        # Use ffmpeg-python for audio extraction
        stream = ffmpeg.input(str(video_path))
        stream = ffmpeg.output(
            stream,
            str(audio_path),
            acodec='pcm_s16le',
            ac=1,
            ar=AudioConfig.sample_rate
        )
        
        # Run asynchronously
        process = await asyncio.create_subprocess_exec(
            'ffmpeg', '-i', str(video_path),
            '-acodec', 'pcm_s16le', '-ac', '1',
            '-ar', str(AudioConfig.sample_rate),
            str(audio_path), '-y',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"Failed to extract audio from {video_path}")
        
        return audio_path
    
    async def _process_video_frames_async(
        self,
        video_path: Path,
        work_dir: Path,
        scene_detection: bool
    ) -> list[Track]:
        """Process video frames and detect faces."""
        # Scene detection
        scenes = []
        if scene_detection:
            scenes = self._detect_scenes(video_path)
        
        # Process video
        cap = cv2.VideoCapture(str(video_path))
        tracks = []
        
        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip if in scene transition
                if self._is_scene_transition(frame_idx, scenes):
                    frame_idx += 1
                    continue
                
                # Detect faces
                detections = await self._detect_faces_async(frame)
                
                # Update tracks
                self.face_tracker.update(detections, frame_idx)
                
                frame_idx += 1
                
                # Yield control periodically
                if frame_idx % 100 == 0:
                    await asyncio.sleep(0)
            
            # Finalize tracks
            tracks = self.face_tracker.get_tracks()
            
        finally:
            cap.release()
        
        return tracks
    
    async def _detect_faces_async(self, frame: np.ndarray) -> list[Detection]:
        """Detect faces in frame asynchronously."""
        # Resize frame for detection
        height, width = frame.shape[:2]
        scale = self.config.facedet_scale
        
        scaled_frame = cv2.resize(
            frame,
            (int(width * scale), int(height * scale))
        )
        
        # Convert to tensor
        img_tensor = torch.from_numpy(scaled_frame).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        # Run detection
        with torch.no_grad():
            detections = self.face_detector(img_tensor)
        
        # Process detections
        face_detections = []
        # TODO: Process raw detections into Detection objects
        
        return face_detections
    
    async def _process_track_async(
        self,
        track: Track,
        audio_path: Path,
        work_dir: Path
    ) -> SyncResult:
        """Process a single face track for synchronization."""
        # Create track directory
        track_dir = work_dir / f"track_{track['start_frame']}"
        track_dir.mkdir(exist_ok=True)
        
        # Extract face crops
        # TODO: Implement face crop extraction
        
        # Run synchronization
        # TODO: Implement synchronization scoring
        
        return SyncResult(
            offset=0,
            confidence=0.0,
            dists=[]
        )
    
    def _detect_scenes(self, video_path: Path) -> list[tuple[int, int]]:
        """Detect scene transitions in video."""
        video_manager = VideoManager([str(video_path)])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())
        
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        
        scenes = scene_manager.get_scene_list()
        video_manager.release()
        
        return [(int(s[0].frame_num), int(s[1].frame_num)) for s in scenes]
    
    def _is_scene_transition(
        self,
        frame_idx: int,
        scenes: list[tuple[int, int]],
        buffer: int = 10
    ) -> bool:
        """Check if frame is near a scene transition."""
        for start, end in scenes:
            if abs(frame_idx - start) < buffer or abs(frame_idx - end) < buffer:
                return True
        return False