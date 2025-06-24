"""Video processing components for SyncNet."""

from pathlib import Path
from typing import Optional, Iterator
import numpy as np
import cv2
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor
import torch

from .base import VideoProcessor
from .config import VideoConfig
from .exceptions import VideoProcessingError, DependencyError
from .logging import get_logger
from .utils import ensure_tensor, Timer, batch_iterator
from .types import BBox


logger = get_logger(__name__)


class OpenCVVideoProcessor(VideoProcessor):
    """OpenCV-based video processor for SyncNet."""
    
    def __init__(self, config: VideoConfig):
        """Initialize video processor.
        
        Args:
            config: Video processing configuration
        """
        self.config = config
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        try:
            import cv2
        except ImportError:
            raise DependencyError(
                "opencv-python", 
                "pip install opencv-contrib-python"
            )
    
    def get_video_info(self, video_path: Path | str) -> dict:
        """Get video metadata.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video metadata
        """
        video_path = Path(video_path)
        
        # Use ffprobe to get accurate metadata
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise VideoProcessingError(video_path, "Failed to get video info")
            
            data = json.loads(result.stdout)
            
            # Extract video stream info
            video_stream = None
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    video_stream = stream
                    break
            
            if not video_stream:
                raise VideoProcessingError(video_path, "No video stream found")
            
            return {
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "fps": eval(video_stream.get("r_frame_rate", "0/1")),
                "duration": float(data["format"].get("duration", 0)),
                "nb_frames": int(video_stream.get("nb_frames", 0)),
                "codec": video_stream.get("codec_name", "unknown")
            }
            
        except (subprocess.SubprocessError, json.JSONDecodeError, KeyError) as e:
            raise VideoProcessingError(video_path, f"Failed to parse video info: {e}")
    
    def extract_frames(
        self, 
        video_path: Path | str, 
        start_frame: int = 0, 
        end_frame: Optional[int] = None
    ) -> list[np.ndarray]:
        """Extract frames from video.
        
        Args:
            video_path: Path to video file
            start_frame: Starting frame index
            end_frame: Ending frame index (None for all frames)
            
        Returns:
            List of video frames as numpy arrays
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise VideoProcessingError(video_path, "Video file not found")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise VideoProcessingError(video_path, "Failed to open video")
        
        try:
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if end_frame is None:
                end_frame = total_frames
            else:
                end_frame = min(end_frame, total_frames)
            
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            with Timer(f"Extracting {end_frame - start_frame} frames", logger):
                for frame_idx in range(start_frame, end_frame):
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(f"Failed to read frame {frame_idx}")
                        break
                    
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            
            logger.info(f"Extracted {len(frames)} frames from {video_path.name}")
            return frames
            
        finally:
            cap.release()
    
    def extract_frames_generator(
        self,
        video_path: Path | str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        batch_size: int = 32
    ) -> Iterator[list[np.ndarray]]:
        """Extract frames in batches for memory efficiency.
        
        Args:
            video_path: Path to video file
            start_frame: Starting frame index
            end_frame: Ending frame index
            batch_size: Number of frames per batch
            
        Yields:
            Batches of frames
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise VideoProcessingError(video_path, "Video file not found")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise VideoProcessingError(video_path, "Failed to open video")
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if end_frame is None:
                end_frame = total_frames
            else:
                end_frame = min(end_frame, total_frames)
            
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            batch = []
            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                batch.append(frame)
                
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            
            # Yield remaining frames
            if batch:
                yield batch
                
        finally:
            cap.release()
    
    def preprocess_frame(self, frame: np.ndarray, bbox: BBox) -> np.ndarray:
        """Preprocess a single video frame.
        
        Args:
            frame: Raw video frame [H, W, C]
            bbox: Face bounding box (x1, y1, x2, y2)
            
        Returns:
            Preprocessed frame [C, H, W]
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Crop face region with padding
        h, w = frame.shape[:2]
        
        # Calculate crop region with padding
        face_w = x2 - x1
        face_h = y2 - y1
        
        pad_w = int(face_w * self.config.crop_scale)
        pad_h = int(face_h * self.config.crop_scale)
        
        crop_x1 = max(0, x1 - pad_w)
        crop_y1 = max(0, y1 - pad_h)
        crop_x2 = min(w, x2 + pad_w)
        crop_y2 = min(h, y2 + pad_h)
        
        # Crop and resize
        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if cropped.size == 0:
            raise VideoProcessingError(
                "frame", 
                f"Invalid crop region: bbox={bbox}, frame_shape={frame.shape}"
            )
        
        # Resize to target size
        resized = cv2.resize(cropped, self.config.frame_size)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert to CHW format
        processed = np.transpose(normalized, (2, 0, 1))
        
        return processed
    
    def prepare_video_batch(
        self,
        frames: list[np.ndarray],
        bboxes: list[BBox],
        sequence_length: int = 5
    ) -> torch.Tensor:
        """Prepare video frames for batch processing.
        
        Args:
            frames: List of video frames
            bboxes: List of face bounding boxes
            sequence_length: Number of frames per sequence
            
        Returns:
            Batched tensor [n_sequences, C, T, H, W]
        """
        if len(frames) != len(bboxes):
            raise ValueError(
                f"Number of frames ({len(frames)}) must match "
                f"number of bboxes ({len(bboxes)})"
            )
        
        if len(frames) < sequence_length:
            raise VideoProcessingError(
                "frames",
                f"Not enough frames ({len(frames)}) for "
                f"sequence length ({sequence_length})"
            )
        
        # Process frames
        processed_frames = []
        for frame, bbox in zip(frames, bboxes):
            processed = self.preprocess_frame(frame, bbox)
            processed_frames.append(processed)
        
        # Create sequences
        sequences = []
        for i in range(len(processed_frames) - sequence_length + 1):
            sequence = processed_frames[i:i + sequence_length]
            # Stack to [C, T, H, W]
            sequence_tensor = np.stack(sequence, axis=1)
            sequences.append(sequence_tensor)
        
        # Stack into batch [N, C, T, H, W]
        batch = np.stack(sequences)
        
        return ensure_tensor(batch)


class ParallelVideoProcessor(OpenCVVideoProcessor):
    """Parallel video processor for faster processing."""
    
    def __init__(self, config: VideoConfig, num_workers: int = 4):
        """Initialize parallel processor.
        
        Args:
            config: Video configuration
            num_workers: Number of parallel workers
        """
        super().__init__(config)
        self.num_workers = num_workers
    
    def extract_frames(
        self, 
        video_path: Path | str, 
        start_frame: int = 0, 
        end_frame: Optional[int] = None
    ) -> list[np.ndarray]:
        """Extract frames in parallel.
        
        Args:
            video_path: Path to video file
            start_frame: Starting frame index
            end_frame: Ending frame index
            
        Returns:
            List of video frames
        """
        video_info = self.get_video_info(video_path)
        total_frames = video_info["nb_frames"]
        
        if end_frame is None:
            end_frame = total_frames
        else:
            end_frame = min(end_frame, total_frames)
        
        num_frames = end_frame - start_frame
        
        # Divide work among workers
        frames_per_worker = num_frames // self.num_workers
        
        def extract_chunk(worker_id: int) -> list[np.ndarray]:
            chunk_start = start_frame + worker_id * frames_per_worker
            
            if worker_id == self.num_workers - 1:
                chunk_end = end_frame
            else:
                chunk_end = chunk_start + frames_per_worker
            
            return super(ParallelVideoProcessor, self).extract_frames(
                video_path, chunk_start, chunk_end
            )
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(extract_chunk, i) 
                for i in range(self.num_workers)
            ]
            
            all_frames = []
            for future in futures:
                chunk_frames = future.result()
                all_frames.extend(chunk_frames)
        
        return all_frames


def create_video_processor(
    config: Optional[VideoConfig] = None,
    parallel: bool = False,
    num_workers: int = 4
) -> VideoProcessor:
    """Factory function to create video processor.
    
    Args:
        config: Video configuration (uses defaults if None)
        parallel: Whether to use parallel processing
        num_workers: Number of workers for parallel processing
        
    Returns:
        Video processor instance
    """
    if config is None:
        config = VideoConfig()
    
    if parallel:
        return ParallelVideoProcessor(config, num_workers)
    else:
        return OpenCVVideoProcessor(config)