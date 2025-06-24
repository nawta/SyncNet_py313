"""Utility functions for SyncNet components."""

from pathlib import Path
from typing import Any, TypeVar, Generator
import torch
import numpy as np
from contextlib import contextmanager
import time
import gc
import os

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from .logging import get_logger


logger = get_logger(__name__)
T = TypeVar('T')


# Memory management utilities
@contextmanager
def torch_memory_manager(device: str = "cuda") -> Generator[None, None, None]:
    """Context manager for PyTorch memory management.
    
    Ensures GPU memory is properly cleaned up after operations.
    """
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
    try:
        yield
    finally:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
            final_memory = torch.cuda.memory_allocated()
            memory_diff = final_memory - initial_memory
            
            if memory_diff > 0:
                logger.debug(f"GPU memory increased by {memory_diff / 1024**2:.2f} MB")


def get_memory_usage() -> dict[str, float]:
    """Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage in MB
    """
    stats = {}
    
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        stats.update({
            "cpu_rss_mb": memory_info.rss / 1024**2,
            "cpu_vms_mb": memory_info.vms / 1024**2,
        })
    
    if torch.cuda.is_available():
        stats.update({
            "gpu_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "gpu_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        })
    
    return stats


# Tensor utilities
def ensure_tensor(
    data: np.ndarray | torch.Tensor, 
    device: str = "cpu",
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Ensure data is a PyTorch tensor on the specified device.
    
    Args:
        data: Input data (numpy array or tensor)
        device: Target device
        dtype: Target data type
        
    Returns:
        PyTorch tensor on specified device
    """
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data).to(dtype)
    elif isinstance(data, torch.Tensor):
        tensor = data.to(dtype)
    else:
        raise TypeError(f"Expected numpy array or torch tensor, got {type(data)}")
    
    return tensor.to(device)


def batch_iterator(
    data: list[T] | np.ndarray | torch.Tensor,
    batch_size: int,
    drop_last: bool = False
) -> Generator[list[T] | np.ndarray | torch.Tensor, None, None]:
    """Iterate over data in batches.
    
    Args:
        data: Input data
        batch_size: Batch size
        drop_last: Whether to drop the last incomplete batch
        
    Yields:
        Batches of data
    """
    n_samples = len(data)
    n_batches = n_samples // batch_size
    
    if not drop_last and n_samples % batch_size != 0:
        n_batches += 1
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        if isinstance(data, list):
            yield data[start_idx:end_idx]
        else:
            yield data[start_idx:end_idx]


# File utilities
def ensure_dir(path: Path | str) -> Path:
    """Ensure directory exists, creating if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_files(
    directory: Path | str,
    pattern: str = "*",
    recursive: bool = True
) -> list[Path]:
    """Find files matching pattern in directory.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    
    if recursive:
        return sorted(directory.rglob(pattern))
    else:
        return sorted(directory.glob(pattern))


# Timing utilities
class Timer:
    """Simple timer for performance measurement."""
    
    def __init__(self, name: str = "Operation", logger: Any = None):
        """Initialize timer.
        
        Args:
            name: Name of operation being timed
            logger: Optional logger instance
        """
        self.name = name
        self.logger = logger or get_logger(__name__)
        self.start_time = None
        self.elapsed = 0.0
    
    def __enter__(self) -> 'Timer':
        """Start timing."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args: Any) -> None:
        """Stop timing and log result."""
        if self.start_time:
            self.elapsed = time.perf_counter() - self.start_time
            self.logger.debug(f"{self.name} took {self.elapsed:.3f}s")


# Validation utilities
def validate_video_path(path: Path | str) -> Path:
    """Validate video file path.
    
    Args:
        path: Video file path
        
    Returns:
        Validated Path object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a video
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
    if path.suffix.lower() not in video_extensions:
        raise ValueError(f"Invalid video format: {path.suffix}")
    
    return path


def validate_audio_path(path: Path | str) -> Path:
    """Validate audio file path.
    
    Args:
        path: Audio file path
        
    Returns:
        Validated Path object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not audio
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    audio_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a'}
    if path.suffix.lower() not in audio_extensions:
        raise ValueError(f"Invalid audio format: {path.suffix}")
    
    return path


# Numerical utilities
def compute_confidence(distances: list[float], method: str = "min_distance") -> float:
    """Compute confidence score from distance values.
    
    Args:
        distances: List of distance/similarity values
        method: Method to compute confidence
        
    Returns:
        Confidence score
    """
    if not distances:
        return 0.0
    
    if method == "min_distance":
        # Lower distance = higher confidence
        min_dist = min(distances)
        # Convert to confidence (assuming distances are typically in range [0, 2])
        confidence = max(0, 2 - min_dist) / 2
    elif method == "mean_distance":
        mean_dist = np.mean(distances)
        confidence = max(0, 2 - mean_dist) / 2
    elif method == "percentile":
        # Use 10th percentile as confidence
        percentile_10 = np.percentile(distances, 10)
        confidence = max(0, 2 - percentile_10) / 2
    else:
        raise ValueError(f"Unknown confidence method: {method}")
    
    return float(confidence)


def normalize_bbox(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int
) -> tuple[float, float, float, float]:
    """Normalize bounding box coordinates to [0, 1] range.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        image_width: Image width
        image_height: Image height
        
    Returns:
        Normalized bounding box
    """
    x1, y1, x2, y2 = bbox
    
    return (
        x1 / image_width,
        y1 / image_height,
        x2 / image_width,
        y2 / image_height
    )


def denormalize_bbox(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int
) -> tuple[int, int, int, int]:
    """Denormalize bounding box from [0, 1] to pixel coordinates.
    
    Args:
        bbox: Normalized bounding box (x1, y1, x2, y2)
        image_width: Image width
        image_height: Image height
        
    Returns:
        Pixel coordinate bounding box
    """
    x1, y1, x2, y2 = bbox
    
    return (
        int(x1 * image_width),
        int(y1 * image_height),
        int(x2 * image_width),
        int(y2 * image_height)
    )


def calculate_iou(
    bbox1: tuple[float, float, float, float],
    bbox2: tuple[float, float, float, float]
) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bounding box (x1, y1, x2, y2)
        bbox2: Second bounding box (x1, y1, x2, y2)
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0