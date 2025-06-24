"""Custom exceptions for SyncNet components."""

from pathlib import Path
from typing import Any


class SyncNetError(Exception):
    """Base exception for all SyncNet errors."""
    pass


class ModelError(SyncNetError):
    """Base exception for model-related errors."""
    pass


class ModelLoadError(ModelError):
    """Raised when model loading fails."""
    
    def __init__(self, path: Path | str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to load model from {path}: {reason}")


class ModelNotFoundError(ModelError):
    """Raised when model file is not found."""
    
    def __init__(self, path: Path | str) -> None:
        self.path = path
        super().__init__(f"Model file not found: {path}")


class InvalidModelError(ModelError):
    """Raised when model structure is invalid."""
    
    def __init__(self, model_type: str, reason: str) -> None:
        self.model_type = model_type
        self.reason = reason
        super().__init__(f"Invalid {model_type} model: {reason}")


class ProcessingError(SyncNetError):
    """Base exception for processing errors."""
    pass


class VideoProcessingError(ProcessingError):
    """Raised when video processing fails."""
    
    def __init__(self, video_path: Path | str, reason: str) -> None:
        self.video_path = video_path
        self.reason = reason
        super().__init__(f"Failed to process video {video_path}: {reason}")


class AudioProcessingError(ProcessingError):
    """Raised when audio processing fails."""
    
    def __init__(self, audio_path: Path | str, reason: str) -> None:
        self.audio_path = audio_path
        self.reason = reason
        super().__init__(f"Failed to process audio {audio_path}: {reason}")


class FaceDetectionError(ProcessingError):
    """Raised when face detection fails."""
    
    def __init__(self, frame_idx: int | None = None, reason: str = "No faces detected") -> None:
        self.frame_idx = frame_idx
        self.reason = reason
        if frame_idx is not None:
            super().__init__(f"Face detection failed at frame {frame_idx}: {reason}")
        else:
            super().__init__(f"Face detection failed: {reason}")


class ValidationError(SyncNetError):
    """Base exception for validation errors."""
    pass


class InputValidationError(ValidationError):
    """Raised when input validation fails."""
    
    def __init__(self, param_name: str, value: Any, expected: str) -> None:
        self.param_name = param_name
        self.value = value
        self.expected = expected
        super().__init__(
            f"Invalid input for '{param_name}': got {type(value).__name__}, "
            f"expected {expected}"
        )


class ConfigurationError(SyncNetError):
    """Raised when configuration is invalid."""
    
    def __init__(self, config_key: str, reason: str) -> None:
        self.config_key = config_key
        self.reason = reason
        super().__init__(f"Invalid configuration for '{config_key}': {reason}")


class DimensionError(ValidationError):
    """Raised when tensor dimensions are incorrect."""
    
    def __init__(self, tensor_name: str, got_shape: tuple, expected_shape: str) -> None:
        self.tensor_name = tensor_name
        self.got_shape = got_shape
        self.expected_shape = expected_shape
        super().__init__(
            f"Invalid dimensions for {tensor_name}: "
            f"got shape {got_shape}, expected {expected_shape}"
        )


class DeviceError(SyncNetError):
    """Raised when device-related operations fail."""
    
    def __init__(self, device: str, reason: str) -> None:
        self.device = device
        self.reason = reason
        super().__init__(f"Device error for '{device}': {reason}")


class DependencyError(SyncNetError):
    """Raised when required dependencies are missing."""
    
    def __init__(self, dependency: str, install_cmd: str | None = None) -> None:
        self.dependency = dependency
        self.install_cmd = install_cmd
        msg = f"Required dependency '{dependency}' is not installed"
        if install_cmd:
            msg += f". Install with: {install_cmd}"
        super().__init__(msg)


class PipelineError(SyncNetError):
    """Base exception for pipeline errors."""
    pass


class PipelineStageError(PipelineError):
    """Raised when a pipeline stage fails."""
    
    def __init__(self, stage_name: str, reason: str, stage_input: Any = None) -> None:
        self.stage_name = stage_name
        self.reason = reason
        self.stage_input = stage_input
        super().__init__(f"Pipeline stage '{stage_name}' failed: {reason}")