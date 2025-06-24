"""Type definitions for SyncNet components."""

from pathlib import Path
from typing import TypedDict
try:
    from typing import TypeAlias, NotRequired
except ImportError:
    # Python 3.9 compatibility
    from typing import Dict, Tuple, List
    TypeAlias = type
    NotRequired = lambda x: x
import numpy as np
import torch

# Basic type aliases
if 'TypeAlias' in globals() and TypeAlias != type:
    BBox: TypeAlias = tuple[float, float, float, float]
    Frame: TypeAlias = np.ndarray
    AudioData: TypeAlias = np.ndarray
    MFCCFeatures: TypeAlias = np.ndarray
else:
    # Python 3.9 compatibility
    BBox = Tuple[float, float, float, float]
    Frame = np.ndarray
    AudioData = np.ndarray
    MFCCFeatures = np.ndarray

# Detection types
class Detection(TypedDict):
    """Face detection result."""
    frame_idx: int
    bbox: BBox
    confidence: float
    landmarks: NotRequired[np.ndarray]

# Track types
class Track(TypedDict):
    """Face track information."""
    frame: np.ndarray
    bbox: np.ndarray
    start_frame: int
    end_frame: int

# Result types
class SyncResult(TypedDict):
    """Synchronization result."""
    offset: int
    confidence: float
    dists: list[float]
    track_id: NotRequired[int]

class PipelineResult(TypedDict):
    """Complete pipeline result."""
    video_path: str | Path
    sync_results: list[SyncResult]
    num_tracks: int
    processing_time: float
    error: NotRequired[str]

# Model types
if 'TypeAlias' in globals() and TypeAlias != type:
    ModelState: TypeAlias = dict[str, torch.Tensor]
    EmbeddingBatch: TypeAlias = torch.Tensor  # Shape: [batch_size, embedding_dim]
else:
    # Python 3.9 compatibility
    ModelState = Dict[str, torch.Tensor]
    EmbeddingBatch = torch.Tensor