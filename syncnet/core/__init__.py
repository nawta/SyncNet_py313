"""Core components for SyncNet.

This module provides the refactored, modern implementation of SyncNet
with clean architecture, proper error handling, and optimizations.
"""

# Base classes and interfaces
from .base import (
    BaseModel,
    AudioEncoder,
    VisualEncoder,
    AVSyncModel,
    FaceDetector,
    AudioProcessor,
    VideoProcessor,
    SyncAnalyzer,
    ModelFactory,
    Pipeline,
)

# Configuration
from .config import (
    ModelConfig,
    FaceDetectorConfig,
    AudioConfig,
    VideoConfig,
    SyncConfig,
    PipelineConfig,
    load_config,
    save_config,
)

# Exceptions
from .exceptions import (
    SyncNetError,
    ModelError,
    ModelLoadError,
    ProcessingError,
    VideoProcessingError,
    AudioProcessingError,
    FaceDetectionError,
    ValidationError,
    ConfigurationError,
)

# Type definitions
from .types import (
    BBox,
    Frame,
    AudioData,
    MFCCFeatures,
    Detection,
    Track,
    SyncResult,
    PipelineResult,
)

# Models
from .models import (
    SyncNetModel,
    SyncNetModelFactory,
    create_syncnet_model,
    load_syncnet_model,
)

# Audio processing
from .audio import (
    MFCCAudioProcessor,
    StreamingAudioProcessor,
    create_audio_processor,
)

# Video processing
from .video import (
    OpenCVVideoProcessor,
    ParallelVideoProcessor,
    create_video_processor,
)

# Synchronization analysis
from .sync_analyzer import (
    SlidingWindowAnalyzer,
    OptimizedSyncAnalyzer,
    CachedSyncAnalyzer,
    create_sync_analyzer,
)

# Utilities
from .utils import (
    torch_memory_manager,
    get_memory_usage,
    ensure_tensor,
    batch_iterator,
    Timer,
    validate_video_path,
    validate_audio_path,
    compute_confidence,
)

# Logging
from .logging import (
    get_logger,
    LoggerManager,
    ProgressLogger,
)

# Legacy compatibility
from .compat import (
    load_legacy_model,
    convert_legacy_config,
)

# Keep backward compatibility
from .inference import SyncNetInstance, InferenceConfig
from .models import save_model, load_model


__all__ = [
    # Base classes
    "BaseModel",
    "AudioEncoder", 
    "VisualEncoder",
    "AVSyncModel",
    "FaceDetector",
    "AudioProcessor",
    "VideoProcessor",
    "SyncAnalyzer",
    "ModelFactory",
    "Pipeline",
    
    # Configuration
    "ModelConfig",
    "FaceDetectorConfig",
    "AudioConfig",
    "VideoConfig",
    "SyncConfig",
    "PipelineConfig",
    "load_config",
    "save_config",
    
    # Exceptions
    "SyncNetError",
    "ModelError",
    "ModelLoadError",
    "ProcessingError",
    "VideoProcessingError",
    "AudioProcessingError",
    "FaceDetectionError",
    "ValidationError",
    "ConfigurationError",
    
    # Types
    "BBox",
    "Frame",
    "AudioData",
    "MFCCFeatures",
    "Detection",
    "Track",
    "SyncResult",
    "PipelineResult",
    
    # Models
    "SyncNetModel",
    "SyncNetModelFactory",
    "create_syncnet_model",
    "load_syncnet_model",
    "save_model",
    "load_model",
    
    # Processors
    "MFCCAudioProcessor",
    "StreamingAudioProcessor",
    "create_audio_processor",
    "OpenCVVideoProcessor",
    "ParallelVideoProcessor",
    "create_video_processor",
    
    # Analyzers
    "SlidingWindowAnalyzer",
    "OptimizedSyncAnalyzer",
    "CachedSyncAnalyzer",
    "create_sync_analyzer",
    
    # Utilities
    "torch_memory_manager",
    "get_memory_usage",
    "ensure_tensor",
    "batch_iterator",
    "Timer",
    "validate_video_path",
    "validate_audio_path",
    "compute_confidence",
    
    # Logging
    "get_logger",
    "LoggerManager",
    "ProgressLogger",
    
    # Compatibility
    "load_legacy_model",
    "convert_legacy_config",
    "SyncNetInstance",
    "InferenceConfig",
]


# Package version
__version__ = "0.1.1"