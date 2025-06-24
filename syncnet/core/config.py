"""Configuration management for SyncNet components."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import json
from abc import ABC, abstractmethod

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from .exceptions import ConfigurationError, ValidationError


@dataclass
class ModelConfig:
    """Configuration for SyncNet models."""
    
    embedding_dim: int = 1024
    audio_channels: int = 12
    visual_channels: int = 5
    device: str = "cuda"
    weights_path: Path | str | None = None
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.embedding_dim <= 0:
            raise ValidationError("embedding_dim", self.embedding_dim, "positive integer")
        if self.audio_channels <= 0:
            raise ValidationError("audio_channels", self.audio_channels, "positive integer")
        if self.visual_channels <= 0:
            raise ValidationError("visual_channels", self.visual_channels, "positive integer")
        if self.device not in ["cpu", "cuda", "mps"]:
            raise ValidationError("device", self.device, "one of ['cpu', 'cuda', 'mps']")


@dataclass
class FaceDetectorConfig:
    """Configuration for face detection."""
    
    model_path: Path | str
    device: str = "cuda"
    confidence_threshold: float = 0.8
    nms_threshold: float = 0.4
    max_faces: int = 10
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not (0 <= self.confidence_threshold <= 1):
            raise ValidationError(
                "confidence_threshold", 
                self.confidence_threshold, 
                "float between 0 and 1"
            )
        if not (0 <= self.nms_threshold <= 1):
            raise ValidationError("nms_threshold", self.nms_threshold, "float between 0 and 1")
        if self.max_faces <= 0:
            raise ValidationError("max_faces", self.max_faces, "positive integer")


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    
    sample_rate: int = 16000
    n_mfcc: int = 12
    window_size: float = 0.025  # seconds
    hop_size: float = 0.010  # seconds
    n_fft: int = 512
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.sample_rate <= 0:
            raise ValidationError("sample_rate", self.sample_rate, "positive integer")
        if self.n_mfcc <= 0:
            raise ValidationError("n_mfcc", self.n_mfcc, "positive integer")
        if self.window_size <= 0:
            raise ValidationError("window_size", self.window_size, "positive float")
        if self.hop_size <= 0:
            raise ValidationError("hop_size", self.hop_size, "positive float")


@dataclass
class VideoConfig:
    """Configuration for video processing."""
    
    fps: float = 25.0
    frame_size: tuple[int, int] = (224, 224)
    crop_scale: float = 0.4
    batch_size: int = 32
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.fps <= 0:
            raise ValidationError("fps", self.fps, "positive float")
        if len(self.frame_size) != 2 or any(s <= 0 for s in self.frame_size):
            raise ValidationError("frame_size", self.frame_size, "tuple of two positive integers")
        if not (0 < self.crop_scale <= 1):
            raise ValidationError("crop_scale", self.crop_scale, "float between 0 and 1")
        if self.batch_size <= 0:
            raise ValidationError("batch_size", self.batch_size, "positive integer")


@dataclass
class SyncConfig:
    """Configuration for synchronization analysis."""
    
    vshift: int = 15  # Maximum video shift in frames
    batch_size: int = 20
    min_track_length: int = 50
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.vshift <= 0:
            raise ValidationError("vshift", self.vshift, "positive integer")
        if self.batch_size <= 0:
            raise ValidationError("batch_size", self.batch_size, "positive integer")
        if self.min_track_length <= 0:
            raise ValidationError("min_track_length", self.min_track_length, "positive integer")


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    face_detector: FaceDetectorConfig | None = None
    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    
    # Processing options
    save_intermediate: bool = False
    output_dir: Path | str | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    num_workers: int = 4
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.num_workers < 0:
            raise ValidationError("num_workers", self.num_workers, "non-negative integer")
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)


class ConfigLoader(ABC):
    """Abstract base class for configuration loaders."""
    
    @abstractmethod
    def load(self, path: Path | str) -> dict[str, Any]:
        """Load configuration from file."""
        pass
    
    @abstractmethod
    def save(self, config: dict[str, Any], path: Path | str) -> None:
        """Save configuration to file."""
        pass


class JSONConfigLoader(ConfigLoader):
    """JSON configuration loader."""
    
    def load(self, path: Path | str) -> dict[str, Any]:
        """Load configuration from JSON file."""
        path = Path(path)
        if not path.exists():
            raise ConfigurationError(str(path), "Configuration file not found")
        
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(str(path), f"Invalid JSON: {e}")
    
    def save(self, config: dict[str, Any], path: Path | str) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2, default=str)


class YAMLConfigLoader(ConfigLoader):
    """YAML configuration loader."""
    
    def __init__(self):
        if not HAS_YAML:
            raise ImportError("PyYAML is required for YAML config support. Install with: pip install pyyaml")
    
    def load(self, path: Path | str) -> dict[str, Any]:
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise ConfigurationError(str(path), "Configuration file not found")
        
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(str(path), f"Invalid YAML: {e}")
    
    def save(self, config: dict[str, Any], path: Path | str) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)


def load_config(path: Path | str) -> PipelineConfig:
    """Load pipeline configuration from file.
    
    Args:
        path: Path to configuration file (JSON or YAML)
        
    Returns:
        Loaded pipeline configuration
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    path = Path(path)
    
    # Select loader based on file extension
    if path.suffix.lower() == '.json':
        loader = JSONConfigLoader()
    elif path.suffix.lower() in ['.yaml', '.yml']:
        loader = YAMLConfigLoader()
    else:
        raise ConfigurationError(
            str(path), 
            "Unsupported file format. Use .json or .yaml"
        )
    
    # Load raw configuration
    config_dict = loader.load(path)
    
    # Parse into dataclasses
    try:
        # Create nested configs
        if 'model' in config_dict:
            config_dict['model'] = ModelConfig(**config_dict['model'])
        if 'face_detector' in config_dict:
            config_dict['face_detector'] = FaceDetectorConfig(**config_dict['face_detector'])
        if 'audio' in config_dict:
            config_dict['audio'] = AudioConfig(**config_dict['audio'])
        if 'video' in config_dict:
            config_dict['video'] = VideoConfig(**config_dict['video'])
        if 'sync' in config_dict:
            config_dict['sync'] = SyncConfig(**config_dict['sync'])
        
        return PipelineConfig(**config_dict)
    except (TypeError, ValueError) as e:
        raise ConfigurationError(str(path), f"Invalid configuration structure: {e}")


def save_config(config: PipelineConfig, path: Path | str) -> None:
    """Save pipeline configuration to file.
    
    Args:
        config: Pipeline configuration to save
        path: Output path (JSON or YAML based on extension)
    """
    path = Path(path)
    
    # Convert to dict
    config_dict = {
        'model': config.model.__dict__,
        'audio': config.audio.__dict__,
        'video': config.video.__dict__,
        'sync': config.sync.__dict__,
        'save_intermediate': config.save_intermediate,
        'log_level': config.log_level,
        'num_workers': config.num_workers,
    }
    
    if config.face_detector:
        config_dict['face_detector'] = config.face_detector.__dict__
    if config.output_dir:
        config_dict['output_dir'] = str(config.output_dir)
    
    # Select loader and save
    if path.suffix.lower() == '.json':
        loader = JSONConfigLoader()
    else:
        loader = YAMLConfigLoader()
    
    loader.save(config_dict, path)