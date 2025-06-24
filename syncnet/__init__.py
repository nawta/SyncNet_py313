"""SyncNet: Audio-Visual Synchronization Detection

A Python 3.13 optimized implementation for detecting lip-sync quality in videos.
"""

__version__ = "0.1.0"
__all__ = ["SyncNetModel", "SyncNetInstance", "SyncNetPipeline"]

from syncnet.core.models import SyncNetModel
from syncnet.core.inference import SyncNetInstance
from syncnet.pipeline.pipeline import SyncNetPipeline