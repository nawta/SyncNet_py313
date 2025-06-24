"""Configuration classes for SyncNet pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

@dataclass(slots=True, kw_only=True)
class PipelineConfig:
    """Configuration for SyncNet pipeline processing."""
    
    # Class constants
    DEFAULT_BATCH_SIZE: ClassVar[int] = 20
    DEFAULT_FRAME_RATE: ClassVar[int] = 25
    
    # Face detection parameters
    facedet_scale: float = 0.25
    crop_scale: float = 0.40
    min_track: int = 50
    num_failed_det: int = 25
    min_face_size: int = 100
    
    # Processing parameters
    frame_rate: int = field(default=DEFAULT_FRAME_RATE)
    batch_size: int = field(default=DEFAULT_BATCH_SIZE)
    vshift: int = 15
    
    # Model weights
    s3fd_weights: Path = field(default_factory=lambda: Path("weights/sfd_face.pth"))
    syncnet_weights: Path = field(default_factory=lambda: Path("weights/syncnet_v2.model"))
    
    # Device configuration
    device: str = "cuda"
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.frame_rate <= 0:
            raise ValueError(f"frame_rate must be positive, got {self.frame_rate}")
        if not 0 < self.facedet_scale <= 1:
            raise ValueError(f"facedet_scale must be in (0, 1], got {self.facedet_scale}")

@dataclass(frozen=True)
class AudioConfig:
    """Configuration for audio processing."""
    
    sample_rate: int = 16000
    window_size: int = 0.025  # 25ms
    step_size: int = 0.010    # 10ms
    num_cepstral: int = 13
    num_features: int = 13

@dataclass(frozen=True)
class VideoConfig:
    """Configuration for video processing."""
    
    fps: int = 25
    crop_size: tuple[int, int] = (224, 224)
    sequence_length: int = 5
    normalization_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalization_std: tuple[float, float, float] = (0.229, 0.224, 0.225)