"""Video processing utilities with Python 3.13 optimizations."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Optional, NamedTuple

import cv2
import numpy as np
import ffmpeg

from syncnet.utils.exceptions import VideoProcessingError, AudioProcessingError


class VideoInfo(NamedTuple):
    """Video metadata information."""
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str


async def extract_audio_async(
    video_path: Path | str,
    output_path: Path | str,
    sample_rate: int = 16000,
    channels: int = 1
) -> Path:
    """Extract audio from video file asynchronously.
    
    Args:
        video_path: Path to input video
        output_path: Path for output audio file
        sample_rate: Target sample rate
        channels: Number of audio channels
        
    Returns:
        Path to extracted audio file
        
    Raises:
        AudioProcessingError: If audio extraction fails
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    
    if not video_path.exists():
        raise AudioProcessingError(
            f"Video file not found: {video_path}",
            audio_path=str(video_path)
        )
    
    try:
        # Build ffmpeg command
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-acodec', 'pcm_s16le',
            '-ac', str(channels),
            '-ar', str(sample_rate),
            str(output_path), '-y'
        ]
        
        # Run asynchronously
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise AudioProcessingError(
                f"FFmpeg failed with return code {process.returncode}",
                audio_path=str(video_path),
                sample_rate=sample_rate,
                error_message=stderr.decode() if stderr else None
            )
        
        return output_path
        
    except Exception as e:
        if isinstance(e, AudioProcessingError):
            raise
        raise AudioProcessingError(
            f"Failed to extract audio: {str(e)}",
            audio_path=str(video_path),
            sample_rate=sample_rate
        ) from e


def extract_audio(
    video_path: Path | str,
    output_path: Path | str,
    sample_rate: int = 16000,
    channels: int = 1
) -> Path:
    """Extract audio from video file (synchronous version).
    
    Args:
        video_path: Path to input video
        output_path: Path for output audio file
        sample_rate: Target sample rate
        channels: Number of audio channels
        
    Returns:
        Path to extracted audio file
    """
    return asyncio.run(
        extract_audio_async(video_path, output_path, sample_rate, channels)
    )


def get_video_info(video_path: Path | str) -> VideoInfo:
    """Get video metadata information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Video metadata
        
    Raises:
        VideoProcessingError: If video cannot be read
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise VideoProcessingError(
            f"Video file not found: {video_path}",
            video_path=str(video_path)
        )
    
    try:
        # Use ffmpeg-python to get metadata
        probe = ffmpeg.probe(str(video_path))
        video_stream = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
            None
        )
        
        if not video_stream:
            raise VideoProcessingError(
                "No video stream found in file",
                video_path=str(video_path)
            )
        
        # Extract metadata
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        
        # Parse frame rate
        fps_parts = video_stream['r_frame_rate'].split('/')
        fps = float(fps_parts[0]) / float(fps_parts[1])
        
        # Get frame count and duration
        frame_count = int(video_stream.get('nb_frames', 0))
        duration = float(video_stream.get('duration', 0))
        
        codec = video_stream.get('codec_name', 'unknown')
        
        return VideoInfo(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            codec=codec
        )
        
    except Exception as e:
        if isinstance(e, VideoProcessingError):
            raise
        raise VideoProcessingError(
            f"Failed to get video info: {str(e)}",
            video_path=str(video_path)
        ) from e


class VideoReader:
    """Efficient video reader with frame caching."""
    
    def __init__(
        self,
        video_path: Path | str,
        cache_size: int = 100
    ) -> None:
        self.video_path = Path(video_path)
        self.cache_size = cache_size
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_cache: dict[int, np.ndarray] = {}
        self.current_frame: int = -1
        
    def __enter__(self) -> VideoReader:
        self.open()
        return self
        
    def __exit__(self, *args) -> None:
        self.close()
    
    def open(self) -> None:
        """Open video file."""
        if not self.video_path.exists():
            raise VideoProcessingError(
                f"Video file not found: {self.video_path}",
                video_path=str(self.video_path)
            )
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise VideoProcessingError(
                f"Failed to open video: {self.video_path}",
                video_path=str(self.video_path)
            )
    
    def close(self) -> None:
        """Close video file."""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.frame_cache.clear()
    
    def read_frame(self, frame_idx: Optional[int] = None) -> Optional[np.ndarray]:
        """Read a specific frame or the next frame.
        
        Args:
            frame_idx: Frame index to read (None for next frame)
            
        Returns:
            Frame array or None if no more frames
        """
        if not self.cap:
            raise VideoProcessingError("Video not opened")
        
        # Check cache first
        if frame_idx is not None and frame_idx in self.frame_cache:
            return self.frame_cache[frame_idx]
        
        # Seek if necessary
        if frame_idx is not None and frame_idx != self.current_frame + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self.current_frame = frame_idx - 1
        
        # Read frame
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        self.current_frame += 1
        
        # Update cache
        if len(self.frame_cache) >= self.cache_size:
            # Remove oldest cached frame
            min_idx = min(self.frame_cache.keys())
            del self.frame_cache[min_idx]
        
        self.frame_cache[self.current_frame] = frame
        
        return frame
    
    def read_frames(
        self,
        start: int,
        end: int,
        step: int = 1
    ) -> list[np.ndarray]:
        """Read a range of frames.
        
        Args:
            start: Start frame index
            end: End frame index (exclusive)
            step: Frame step
            
        Returns:
            List of frames
        """
        frames = []
        for idx in range(start, end, step):
            frame = self.read_frame(idx)
            if frame is None:
                break
            frames.append(frame)
        return frames
    
    @property
    def fps(self) -> float:
        """Get video frame rate."""
        if not self.cap:
            raise VideoProcessingError("Video not opened")
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    @property
    def frame_count(self) -> int:
        """Get total number of frames."""
        if not self.cap:
            raise VideoProcessingError("Video not opened")
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    @property
    def width(self) -> int:
        """Get video width."""
        if not self.cap:
            raise VideoProcessingError("Video not opened")
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    @property
    def height(self) -> int:
        """Get video height."""
        if not self.cap:
            raise VideoProcessingError("Video not opened")
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))