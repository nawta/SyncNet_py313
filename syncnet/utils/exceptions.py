"""Custom exceptions for SyncNet with Python 3.13 exception groups."""

from __future__ import annotations

from typing import Optional


class SyncNetError(Exception):
    """Base exception for all SyncNet errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.details = details or {}


class ModelLoadError(SyncNetError):
    """Error loading model weights or architecture."""
    pass


class VideoProcessingError(SyncNetError):
    """Error processing video files."""
    
    def __init__(
        self, 
        message: str,
        video_path: Optional[str] = None,
        frame_number: Optional[int] = None,
        **kwargs
    ) -> None:
        details = {"video_path": video_path, "frame_number": frame_number}
        details.update(kwargs)
        super().__init__(message, details)


class AudioProcessingError(SyncNetError):
    """Error processing audio data."""
    
    def __init__(
        self,
        message: str,
        audio_path: Optional[str] = None,
        sample_rate: Optional[int] = None,
        **kwargs
    ) -> None:
        details = {"audio_path": audio_path, "sample_rate": sample_rate}
        details.update(kwargs)
        super().__init__(message, details)


class FaceDetectionError(SyncNetError):
    """Error in face detection process."""
    
    def __init__(
        self,
        message: str,
        num_faces: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        **kwargs
    ) -> None:
        details = {
            "num_faces": num_faces,
            "confidence_threshold": confidence_threshold
        }
        details.update(kwargs)
        super().__init__(message, details)


class PipelineError(SyncNetError):
    """Error in pipeline execution with support for exception groups."""
    
    def __init__(
        self,
        message: str,
        errors: Optional[list[Exception]] = None,
        stage: Optional[str] = None,
        **kwargs
    ) -> None:
        details = {"stage": stage, "num_errors": len(errors) if errors else 0}
        details.update(kwargs)
        super().__init__(message, details)
        self.errors = errors or []
    
    def raise_group(self) -> None:
        """Raise as exception group in Python 3.13."""
        if self.errors:
            # Python 3.13 syntax for exception groups
            raise ExceptionGroup(str(self), self.errors)