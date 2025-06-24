"""Base classes and interfaces for SyncNet components."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, TypeVar, Generic
import torch
import numpy as np
from torch import nn

from .types import BBox, Detection, Track, SyncResult, ModelState


T = TypeVar('T')
ModelType = TypeVar('ModelType', bound=nn.Module)


class BaseModel(ABC, nn.Module):
    """Abstract base class for all SyncNet models."""
    
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass of the model."""
        pass
    
    def save_weights(self, path: Path | str) -> None:
        """Save model weights to disk."""
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path: Path | str, device: str = "cpu") -> None:
        """Load model weights from disk."""
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)


class AudioEncoder(BaseModel):
    """Abstract base class for audio encoders."""
    
    @abstractmethod
    def forward_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio features.
        
        Args:
            audio: Audio tensor of shape [batch, channels, time, freq]
            
        Returns:
            Audio embeddings of shape [batch, embedding_dim]
        """
        pass


class VisualEncoder(BaseModel):
    """Abstract base class for visual encoders."""
    
    @abstractmethod
    def forward_visual(self, visual: torch.Tensor) -> torch.Tensor:
        """Encode visual features.
        
        Args:
            visual: Visual tensor of shape [batch, channels, frames, height, width]
            
        Returns:
            Visual embeddings of shape [batch, embedding_dim]
        """
        pass


class AVSyncModel(AudioEncoder, VisualEncoder):
    """Abstract base class for audio-visual synchronization models."""
    
    @abstractmethod
    def compute_similarity(
        self, 
        audio_embeddings: torch.Tensor, 
        visual_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity between audio and visual embeddings.
        
        Args:
            audio_embeddings: Audio embeddings [batch, embedding_dim]
            visual_embeddings: Visual embeddings [batch, embedding_dim]
            
        Returns:
            Similarity scores [batch]
        """
        pass


class FaceDetector(ABC):
    """Abstract base class for face detectors."""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect faces in an image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            List of face detections
        """
        pass
    
    @abstractmethod
    def batch_detect(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """Detect faces in multiple images.
        
        Args:
            images: List of input images
            
        Returns:
            List of detection lists for each image
        """
        pass


class FaceTrackerProtocol(Protocol):
    """Protocol for face tracking implementations."""
    
    def track(self, detections: list[list[Detection]]) -> list[Track]:
        """Track faces across multiple frames.
        
        Args:
            detections: List of detections per frame
            
        Returns:
            List of face tracks
        """
        ...


class AudioProcessor(ABC):
    """Abstract base class for audio processors."""
    
    @abstractmethod
    def extract_features(self, audio_path: Path | str) -> np.ndarray:
        """Extract audio features from file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio features array
        """
        pass
    
    @abstractmethod
    def preprocess(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Preprocess raw audio data.
        
        Args:
            audio: Raw audio samples
            sample_rate: Audio sample rate
            
        Returns:
            Preprocessed audio features
        """
        pass


class VideoProcessor(ABC):
    """Abstract base class for video processors."""
    
    @abstractmethod
    def extract_frames(
        self, 
        video_path: Path | str, 
        start_frame: int = 0, 
        end_frame: int | None = None
    ) -> list[np.ndarray]:
        """Extract frames from video.
        
        Args:
            video_path: Path to video file
            start_frame: Starting frame index
            end_frame: Ending frame index (None for all frames)
            
        Returns:
            List of video frames
        """
        pass
    
    @abstractmethod
    def preprocess_frame(self, frame: np.ndarray, bbox: BBox) -> np.ndarray:
        """Preprocess a single video frame.
        
        Args:
            frame: Raw video frame
            bbox: Face bounding box
            
        Returns:
            Preprocessed frame
        """
        pass


class SyncAnalyzer(ABC):
    """Abstract base class for synchronization analyzers."""
    
    @abstractmethod
    def analyze(
        self, 
        audio_features: torch.Tensor, 
        visual_features: torch.Tensor
    ) -> SyncResult:
        """Analyze audio-visual synchronization.
        
        Args:
            audio_features: Audio feature tensor
            visual_features: Visual feature tensor
            
        Returns:
            Synchronization analysis result
        """
        pass


class ModelFactory(ABC, Generic[ModelType]):
    """Abstract factory for creating models."""
    
    @abstractmethod
    def create_model(self, config: dict[str, Any]) -> ModelType:
        """Create a model instance.
        
        Args:
            config: Model configuration
            
        Returns:
            Model instance
        """
        pass
    
    @abstractmethod
    def load_model(self, path: Path | str, device: str = "cpu") -> ModelType:
        """Load a model from disk.
        
        Args:
            path: Path to model weights
            device: Device to load model on
            
        Returns:
            Loaded model instance
        """
        pass


class Pipeline(ABC, Generic[T]):
    """Abstract base class for processing pipelines."""
    
    @abstractmethod
    def process(self, input_data: T) -> Any:
        """Process input data through the pipeline.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processing result
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_data: T) -> None:
        """Validate input data.
        
        Args:
            input_data: Input data to validate
            
        Raises:
            ValueError: If input is invalid
        """
        pass