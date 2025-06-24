"""SyncNet Python: Audio-visual synchronization detection.

This package provides a PyTorch implementation of SyncNet for detecting
synchronization between audio and video in multimedia content.
"""

__version__ = "0.1.1"

# Import main components
try:
    from .syncnet_pipeline import SyncNetPipeline
    from .SyncNetModel import S as SyncNetModel
    from .SyncNetInstance import SyncNetInstance
except ImportError:
    # Fallback for development
    SyncNetPipeline = None
    SyncNetModel = None
    SyncNetInstance = None

__all__ = [
    "SyncNetPipeline",
    "SyncNetModel", 
    "SyncNetInstance",
    "__version__"
]

def get_version():
    """Get package version."""
    return __version__