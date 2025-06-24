"""Compatibility layer for loading original SyncNet models."""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path

from syncnet.core.models import SyncNetModel
from syncnet.core.config import ModelConfig


class LegacySyncNetModel(nn.Module):
    """Legacy SyncNet model structure for compatibility."""
    
    def __init__(self, num_layers_in_fc_layers: int = 1024) -> None:
        super().__init__()
        
        # Audio encoder (legacy names)
        self.netcnnaud = nn.Sequential(
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
        
        # Audio FC (legacy names)
        self.netfcaud = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )
        
        # Visual encoder (legacy names)
        self.netcnnlip = nn.Sequential(
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
        
        # Visual FC (legacy names)
        self.netfclip = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )
    
    def forward_aud(self, x: torch.Tensor) -> torch.Tensor:
        """Legacy audio forward."""
        mid = self.netcnnaud(x)
        mid = mid.view((mid.size()[0], -1))
        out = self.netfcaud(mid)
        return out
    
    def forward_lip(self, x: torch.Tensor) -> torch.Tensor:
        """Legacy visual forward."""
        mid = self.netcnnlip(x)
        mid = mid.view((mid.size()[0], -1))
        out = self.netfclip(mid)
        return out
    
    def forward_lipfeat(self, x: torch.Tensor) -> torch.Tensor:
        """Legacy visual features forward."""
        mid = self.netcnnlip(x)
        out = mid.view((mid.size()[0], -1))
        return out


def load_legacy_model(path: Path | str, device: str = "cpu") -> SyncNetModel:
    """Load legacy SyncNet model and convert to new format.
    
    Args:
        path: Path to legacy model file
        device: Device to load model on
        
    Returns:
        Converted SyncNetModel instance
    """
    # Load the full legacy model
    legacy_model = torch.load(path, map_location=device)
    
    # Create new model
    config = ModelConfig(embedding_dim=1024, device=device)
    new_model = SyncNetModel(config)
    
    # Map legacy state to new model
    if hasattr(legacy_model, 'state_dict'):
        legacy_state = legacy_model.state_dict()
    else:
        # Handle case where the saved file is just the state dict
        legacy_state = legacy_model
    
    # Create mapping between old and new parameter names
    new_state = new_model.state_dict()
    
    # Map audio encoder parameters
    for i, (old_key, new_key) in enumerate([
        ('netcnnaud.0', 'audio_encoder.0'),   # Conv2d
        ('netcnnaud.1', 'audio_encoder.1'),   # BatchNorm2d
        ('netcnnaud.2', 'audio_encoder.2'),   # ReLU
        ('netcnnaud.3', 'audio_encoder.3'),   # MaxPool2d
        ('netcnnaud.4', 'audio_encoder.4'),   # Conv2d
        ('netcnnaud.5', 'audio_encoder.5'),   # BatchNorm2d
        ('netcnnaud.6', 'audio_encoder.6'),   # ReLU
        ('netcnnaud.7', 'audio_encoder.7'),   # MaxPool2d
        ('netcnnaud.8', 'audio_encoder.8'),   # Conv2d
        ('netcnnaud.9', 'audio_encoder.9'),   # BatchNorm2d
        ('netcnnaud.10', 'audio_encoder.10'), # ReLU
        ('netcnnaud.11', 'audio_encoder.11'), # Conv2d
        ('netcnnaud.12', 'audio_encoder.12'), # BatchNorm2d
        ('netcnnaud.13', 'audio_encoder.13'), # ReLU
        ('netcnnaud.14', 'audio_encoder.14'), # Conv2d
        ('netcnnaud.15', 'audio_encoder.15'), # BatchNorm2d
        ('netcnnaud.16', 'audio_encoder.16'), # ReLU
        ('netcnnaud.17', 'audio_encoder.17'), # MaxPool2d
        ('netcnnaud.18', 'audio_encoder.18'), # Conv2d
        ('netcnnaud.19', 'audio_encoder.19'), # BatchNorm2d
        ('netcnnaud.20', 'audio_encoder.20'), # ReLU
    ]):
        for param in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
            old_param = f"{old_key}.{param}"
            new_param = f"{new_key}.{param}"
            if old_param in legacy_state and new_param in new_state:
                new_state[new_param] = legacy_state[old_param]
    
    # Map audio FC parameters
    for i, (old_key, new_key) in enumerate([
        ('netfcaud.0', 'audio_fc.0'),  # Linear
        ('netfcaud.1', 'audio_fc.1'),  # BatchNorm1d
        ('netfcaud.2', 'audio_fc.2'),  # ReLU
        ('netfcaud.3', 'audio_fc.3'),  # Linear
    ]):
        for param in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
            old_param = f"{old_key}.{param}"
            new_param = f"{new_key}.{param}"
            if old_param in legacy_state and new_param in new_state:
                new_state[new_param] = legacy_state[old_param]
    
    # Map visual encoder parameters
    for i, (old_key, new_key) in enumerate([
        ('netcnnlip.0', 'visual_encoder.0'),   # Conv3d
        ('netcnnlip.1', 'visual_encoder.1'),   # BatchNorm3d
        ('netcnnlip.2', 'visual_encoder.2'),   # ReLU
        ('netcnnlip.3', 'visual_encoder.3'),   # MaxPool3d
        ('netcnnlip.4', 'visual_encoder.4'),   # Conv3d
        ('netcnnlip.5', 'visual_encoder.5'),   # BatchNorm3d
        ('netcnnlip.6', 'visual_encoder.6'),   # ReLU
        ('netcnnlip.7', 'visual_encoder.7'),   # MaxPool3d
        ('netcnnlip.8', 'visual_encoder.8'),   # Conv3d
        ('netcnnlip.9', 'visual_encoder.9'),   # BatchNorm3d
        ('netcnnlip.10', 'visual_encoder.10'), # ReLU
        ('netcnnlip.11', 'visual_encoder.11'), # Conv3d
        ('netcnnlip.12', 'visual_encoder.12'), # BatchNorm3d
        ('netcnnlip.13', 'visual_encoder.13'), # ReLU
        ('netcnnlip.14', 'visual_encoder.14'), # Conv3d
        ('netcnnlip.15', 'visual_encoder.15'), # BatchNorm3d
        ('netcnnlip.16', 'visual_encoder.16'), # ReLU
        ('netcnnlip.17', 'visual_encoder.17'), # MaxPool3d
        ('netcnnlip.18', 'visual_encoder.18'), # Conv3d
        ('netcnnlip.19', 'visual_encoder.19'), # BatchNorm3d
        ('netcnnlip.20', 'visual_encoder.20'), # ReLU
    ]):
        for param in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
            old_param = f"{old_key}.{param}"
            new_param = f"{new_key}.{param}"
            if old_param in legacy_state and new_param in new_state:
                new_state[new_param] = legacy_state[old_param]
    
    # Map visual FC parameters
    for i, (old_key, new_key) in enumerate([
        ('netfclip.0', 'visual_fc.0'),  # Linear
        ('netfclip.1', 'visual_fc.1'),  # BatchNorm1d
        ('netfclip.2', 'visual_fc.2'),  # ReLU
        ('netfclip.3', 'visual_fc.3'),  # Linear
    ]):
        for param in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
            old_param = f"{old_key}.{param}"
            new_param = f"{new_key}.{param}"
            if old_param in legacy_state and new_param in new_state:
                new_state[new_param] = legacy_state[old_param]
    
    # Load the mapped state
    new_model.load_state_dict(new_state)
    
    return new_model


def convert_legacy_config(legacy_config: dict) -> dict:
    """Convert legacy configuration to new format.
    
    Args:
        legacy_config: Legacy configuration dictionary
        
    Returns:
        New format configuration dictionary
    """
    new_config = {
        'model': {
            'embedding_dim': legacy_config.get('num_layers_in_fc_layers', 1024),
            'device': legacy_config.get('device', 'cuda'),
        },
        'audio': {
            'sample_rate': legacy_config.get('sample_rate', 16000),
            'n_mfcc': legacy_config.get('mfcc_num', 12),
        },
        'video': {
            'fps': legacy_config.get('fps', 25.0),
            'frame_size': (224, 224),
            'crop_scale': legacy_config.get('crop_scale', 0.4),
        },
        'sync': {
            'vshift': legacy_config.get('vshift', 15),
            'batch_size': legacy_config.get('batch_size', 20),
            'min_track_length': legacy_config.get('min_track', 50),
        }
    }
    
    return new_config