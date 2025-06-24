"""S3FD face detector with Python 3.13 optimizations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

from syncnet.detectors.s3fd.utils import Detect, PriorBox
from syncnet.core.types import BBox, Detection


class L2Norm(nn.Module):
    """L2 normalization layer."""
    
    def __init__(self, n_channels: int, scale: float = 1.0) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.gamma = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize parameters."""
        init.constant_(self.weight, self.gamma)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply L2 normalization."""
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class S3FDNet(nn.Module):
    """S3FD: Single Shot Scale-invariant Face Detector.
    
    Args:
        device: Device to run the model on
    """
    
    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # VGG16 backbone layers
        self.vgg_layers = self._build_vgg_layers()
        
        # L2 normalization layers
        self.l2_norm_conv3_3 = L2Norm(256, 10)
        self.l2_norm_conv4_3 = L2Norm(512, 8)
        self.l2_norm_conv5_3 = L2Norm(512, 5)
        
        # Extra feature layers
        self.extras = self._build_extra_layers()
        
        # Detection head layers
        self.loc_layers = nn.ModuleList([
            nn.Conv2d(256, 4, 3, padding=1),
            nn.Conv2d(512, 4, 3, padding=1),
            nn.Conv2d(512, 4, 3, padding=1),
            nn.Conv2d(1024, 4, 3, padding=1),
            nn.Conv2d(512, 4, 3, padding=1),
            nn.Conv2d(256, 4, 3, padding=1),
        ])
        
        self.conf_layers = nn.ModuleList([
            nn.Conv2d(256, 4, 3, padding=1),    # conv3_3
            nn.Conv2d(512, 2, 3, padding=1),    # conv4_3
            nn.Conv2d(512, 2, 3, padding=1),    # conv5_3
            nn.Conv2d(1024, 2, 3, padding=1),   # fc7
            nn.Conv2d(512, 2, 3, padding=1),    # conv6_2
            nn.Conv2d(256, 2, 3, padding=1),    # conv7_2
        ])
        
        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect()
    
    def _build_vgg_layers(self) -> nn.ModuleList:
        """Build VGG16 backbone layers."""
        layers = []
        in_channels = 3
        
        # VGG16 configuration
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers.extend([conv2d, nn.ReLU(inplace=True)])
                in_channels = v
        
        # Convert fc6 and fc7 to convolutional layers
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        
        layers.extend([
            pool5,
            conv6, nn.ReLU(inplace=True),
            conv7, nn.ReLU(inplace=True)
        ])
        
        return nn.ModuleList(layers)
    
    def _build_extra_layers(self) -> nn.ModuleList:
        """Build extra feature layers."""
        layers = []
        
        # conv6_1 and conv6_2
        layers.extend([
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        ])
        
        # conv7_1 and conv7_2
        layers.extend([
            nn.Conv2d(512, 128, kernel_size=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        ])
        
        return nn.ModuleList(layers)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            x: Input tensor [batch, 3, height, width]
            
        Returns:
            Tuple of (locations, confidences, prior_boxes)
        """
        sources = []
        loc = []
        conf = []
        
        # Apply VGG layers
        for k in range(16):
            x = self.vgg_layers[k](x)
        
        # conv3_3
        s = self.l2_norm_conv3_3(x)
        sources.append(s)
        
        # Apply more VGG layers
        for k in range(16, 23):
            x = self.vgg_layers[k](x)
        
        # conv4_3
        s = self.l2_norm_conv4_3(x)
        sources.append(s)
        
        # Apply more VGG layers
        for k in range(23, 30):
            x = self.vgg_layers[k](x)
        
        # conv5_3
        s = self.l2_norm_conv5_3(x)
        sources.append(s)
        
        # Apply remaining VGG layers (fc6, fc7)
        for k in range(30, len(self.vgg_layers)):
            x = self.vgg_layers[k](x)
        
        # fc7
        sources.append(x)
        
        # Apply extra layers
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        
        # Apply detection layers
        for (x, l, c) in zip(sources, self.loc_layers, self.conf_layers):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, 2),
            None  # Prior boxes will be generated separately
        )
        
        return output
    
    def load_weights(self, path: Path | str) -> None:
        """Load pre-trained weights.
        
        Args:
            path: Path to weights file
        """
        state_dict = torch.load(path, map_location=self.device)
        
        # Map old parameter names to new ones
        mapped_state = {}
        for key, value in state_dict.items():
            # Map vgg -> vgg_layers
            if key.startswith('vgg.'):
                new_key = key.replace('vgg.', 'vgg_layers.')
                mapped_state[new_key] = value
            # Map L2Norm3_3 -> l2_norm_conv3_3
            elif key == 'L2Norm3_3.weight':
                mapped_state['l2_norm_conv3_3.weight'] = value
            elif key == 'L2Norm4_3.weight':
                mapped_state['l2_norm_conv4_3.weight'] = value
            elif key == 'L2Norm5_3.weight':
                mapped_state['l2_norm_conv5_3.weight'] = value
            # Map loc -> loc_layers
            elif key.startswith('loc.'):
                new_key = key.replace('loc.', 'loc_layers.')
                mapped_state[new_key] = value
            # Map conf -> conf_layers
            elif key.startswith('conf.'):
                new_key = key.replace('conf.', 'conf_layers.')
                mapped_state[new_key] = value
            else:
                mapped_state[key] = value
        
        self.load_state_dict(mapped_state, strict=False)