"""SyncNet model implementation with clean architecture."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from torch import Tensor

from .base import AVSyncModel, ModelFactory
from .config import ModelConfig
from .exceptions import ModelLoadError, DimensionError
from .logging import get_logger
from .types import ModelState


logger = get_logger(__name__)


class AudioEncoderNetwork(nn.Module):
    """Audio encoder network for SyncNet."""
    
    def __init__(self, embedding_dim: int = 1024) -> None:
        super().__init__()
        
        # 2D CNN for spectrograms
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
            
            nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),
            
            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            
            nn.Conv2d(256, 512, kernel_size=(5, 4), padding=(0, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # FC layers
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        # Validate input dimensions
        if x.dim() != 4:
            raise DimensionError("audio input", x.shape, "[batch, 1, height, width]")
        
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        return self.fc_layers(features)


class VisualEncoderNetwork(nn.Module):
    """Visual encoder network for SyncNet."""
    
    def __init__(self, embedding_dim: int = 1024) -> None:
        super().__init__()
        
        # 3D CNN for video sequences
        self.conv_layers = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=0),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            
            nn.Conv3d(96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            
            nn.Conv3d(256, 512, kernel_size=(1, 6, 6), padding=0),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )
        
        # FC layers
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        # Validate input dimensions
        if x.dim() != 5:
            raise DimensionError("visual input", x.shape, "[batch, 3, frames, height, width]")
        
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        return self.fc_layers(features)
    
    def extract_features(self, x: Tensor) -> Tensor:
        """Extract features without FC layers."""
        if x.dim() != 5:
            raise DimensionError("visual input", x.shape, "[batch, 3, frames, height, width]")
        
        features = self.conv_layers(x)
        return features.view(features.size(0), -1)


class SyncNetModel(AVSyncModel):
    """SyncNet model for audio-visual synchronization.
    
    This model processes audio and visual features separately through
    dedicated encoders and produces embeddings for synchronization scoring.
    """
    
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        
        # Initialize encoders
        self.audio_encoder = AudioEncoderNetwork(config.embedding_dim)
        self.visual_encoder = VisualEncoderNetwork(config.embedding_dim)
        
        logger.info(f"Initialized SyncNet model with embedding_dim={config.embedding_dim}")
    
    def forward_audio(self, audio: Tensor) -> Tensor:
        """Encode audio features."""
        return self.audio_encoder(audio)
    
    def forward_visual(self, visual: Tensor) -> Tensor:
        """Encode visual features."""
        return self.visual_encoder(visual)
    
    def compute_similarity(
        self, 
        audio_embeddings: Tensor, 
        visual_embeddings: Tensor
    ) -> Tensor:
        """Compute cosine similarity between embeddings."""
        # L2 normalize embeddings
        audio_norm = nn.functional.normalize(audio_embeddings, p=2, dim=1)
        visual_norm = nn.functional.normalize(visual_embeddings, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.sum(audio_norm * visual_norm, dim=1)
        
        return similarity
    
    def forward(self, audio: Tensor, visual: Tensor) -> Tensor:
        """Forward pass computing similarity."""
        audio_emb = self.forward_audio(audio)
        visual_emb = self.forward_visual(visual)
        return self.compute_similarity(audio_emb, visual_emb)
    
    def extract_visual_features(self, visual: Tensor) -> Tensor:
        """Extract visual features without FC layers."""
        return self.visual_encoder.extract_features(visual)


class SyncNetModelFactory(ModelFactory[SyncNetModel]):
    """Factory for creating SyncNet models."""
    
    def create_model(self, config: dict[str, Any] | ModelConfig) -> SyncNetModel:
        """Create a new SyncNet model."""
        if isinstance(config, dict):
            config = ModelConfig(**config)
        
        logger.info("Creating new SyncNet model")
        return SyncNetModel(config)
    
    def load_model(self, path: Path | str, device: str = "cpu") -> SyncNetModel:
        """Load a SyncNet model from disk."""
        path = Path(path)
        
        if not path.exists():
            raise ModelLoadError(path, "Model file not found")
        
        try:
            logger.info(f"Loading SyncNet model from {path}")
            
            # Load state dict
            state_dict = torch.load(path, map_location=device)
            
            # Infer config from state dict
            # Check embedding dimension from FC layer
            fc_weight_key = None
            for key in state_dict.keys():
                if 'fc_layers.3.weight' in key:
                    fc_weight_key = key
                    break
            
            if fc_weight_key:
                embedding_dim = state_dict[fc_weight_key].shape[0]
            else:
                embedding_dim = 1024  # Default
            
            # Create model with inferred config
            config = ModelConfig(embedding_dim=embedding_dim, device=device)
            model = self.create_model(config)
            
            # Load weights
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            
            logger.info(f"Model loaded successfully on {device}")
            return model
            
        except Exception as e:
            raise ModelLoadError(path, str(e))


# Convenience functions
def create_syncnet_model(
    embedding_dim: int = 1024,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> SyncNetModel:
    """Create a new SyncNet model with default configuration."""
    config = ModelConfig(embedding_dim=embedding_dim, device=device)
    factory = SyncNetModelFactory()
    return factory.create_model(config)


def load_syncnet_model(
    path: Path | str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> SyncNetModel:
    """Load a SyncNet model from file."""
    factory = SyncNetModelFactory()
    return factory.load_model(path, device)


# Legacy compatibility functions
def save_model(model: nn.Module, path: Path | str) -> None:
    """Save a PyTorch model to file (legacy compatibility).
    
    Args:
        model: Model to save
        path: Path to save the model
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        torch.save(model, f)
    logger.info(f"{path} saved.")


def load_model(path: Path | str) -> nn.Module:
    """Load a PyTorch model from file (legacy compatibility).
    
    Args:
        path: Path to the saved model
        
    Returns:
        Loaded PyTorch model
    """
    return torch.load(path, map_location='cpu')