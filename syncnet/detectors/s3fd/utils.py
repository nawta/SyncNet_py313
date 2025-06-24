"""Utility functions for S3FD face detector."""

from __future__ import annotations

from itertools import product
from math import sqrt
from typing import Optional

import torch
import numpy as np

from syncnet.core.types import BBox, Detection


class PriorBox:
    """Generate prior boxes for S3FD.
    
    Args:
        image_size: Input image size
        feature_maps: Feature map sizes for each detection layer
        min_sizes: Minimum box sizes for each feature map
        steps: Feature map strides
        clip: Whether to clip boxes to image boundaries
    """
    
    def __init__(
        self,
        image_size: tuple[int, int] = (300, 300),
        feature_maps: Optional[list[int]] = None,
        min_sizes: Optional[list[int]] = None,
        steps: Optional[list[int]] = None,
        clip: bool = True
    ) -> None:
        self.image_size = image_size
        self.feature_maps = feature_maps or [38, 19, 10, 5, 3, 1]
        self.min_sizes = min_sizes or [16, 32, 64, 128, 256, 512]
        self.steps = steps or [8, 16, 32, 64, 100, 300]
        self.clip = clip
    
    def forward(self) -> torch.Tensor:
        """Generate prior boxes.
        
        Returns:
            Prior boxes tensor [num_priors, 4]
        """
        mean = []
        
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size[0] / self.steps[k]
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                
                s_k = self.min_sizes[k] / self.image_size[0]
                mean.extend([cx, cy, s_k, s_k])
        
        output = torch.Tensor(mean).view(-1, 4)
        
        if self.clip:
            output.clamp_(max=1, min=0)
        
        return output


class Detect:
    """Decode predictions into detections.
    
    Args:
        num_classes: Number of classes
        bkg_label: Background label index
        conf_thresh: Confidence threshold
        nms_thresh: NMS threshold
        top_k: Keep top-k detections per class
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        bkg_label: int = 0,
        conf_thresh: float = 0.05,
        nms_thresh: float = 0.3,
        top_k: int = 750
    ) -> None:
        self.num_classes = num_classes
        self.bkg_label = bkg_label
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.top_k = top_k
    
    def forward(
        self,
        loc_data: torch.Tensor,
        conf_data: torch.Tensor,
        prior_data: torch.Tensor
    ) -> list[list[Detection]]:
        """Apply detection decoding and NMS.
        
        Args:
            loc_data: Location predictions [batch, num_priors, 4]
            conf_data: Confidence predictions [batch, num_priors, num_classes]
            prior_data: Prior boxes [num_priors, 4]
            
        Returns:
            List of detections for each image in batch
        """
        batch_size = loc_data.size(0)
        num_priors = prior_data.size(0)
        
        # Decode predictions
        boxes = self._decode(loc_data.view(-1, 4), prior_data)
        boxes = boxes.view(batch_size, num_priors, 4)
        
        # Apply softmax to confidences
        conf_preds = conf_data.view(batch_size, num_priors, self.num_classes)
        
        output = []
        
        for i in range(batch_size):
            decoded_boxes = boxes[i].clone()
            conf_scores = conf_preds[i].clone()
            
            # For each class except background
            detections = []
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[:, cl] > self.conf_thresh
                scores = conf_scores[c_mask, cl]
                
                if scores.size(0) == 0:
                    continue
                
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes_cl = decoded_boxes[l_mask].view(-1, 4)
                
                # Apply NMS
                ids, count = self._nms(boxes_cl, scores, self.nms_thresh, self.top_k)
                
                # Convert to detections
                for j in range(count):
                    idx = ids[j]
                    det = Detection(
                        frame_idx=i,
                        bbox=(
                            float(boxes_cl[idx, 0]),
                            float(boxes_cl[idx, 1]),
                            float(boxes_cl[idx, 2]),
                            float(boxes_cl[idx, 3])
                        ),
                        confidence=float(scores[idx])
                    )
                    detections.append(det)
            
            output.append(detections)
        
        return output
    
    def _decode(
        self,
        loc: torch.Tensor,
        priors: torch.Tensor,
        variances: list[float] = [0.1, 0.2]
    ) -> torch.Tensor:
        """Decode location predictions using priors.
        
        Args:
            loc: Location predictions [num_priors, 4]
            priors: Prior boxes [num_priors, 4]
            variances: Variances for decoding
            
        Returns:
            Decoded bounding boxes [num_priors, 4]
        """
        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])
        ), 1)
        
        # Convert to [xmin, ymin, xmax, ymax]
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        
        return boxes
    
    def _nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        overlap: float = 0.5,
        top_k: int = 200
    ) -> tuple[torch.Tensor, int]:
        """Apply non-maximum suppression.
        
        Args:
            boxes: Bounding boxes [num_boxes, 4]
            scores: Confidence scores [num_boxes]
            overlap: IoU threshold for suppression
            top_k: Keep top-k boxes
            
        Returns:
            Tuple of (kept indices, number of kept boxes)
        """
        keep = scores.new_zeros(scores.size(0), dtype=torch.long)
        
        if boxes.numel() == 0:
            return keep, 0
        
        # Sort by score
        v, idx = scores.sort(0, descending=True)
        
        # Keep top-k
        idx = idx[:top_k]
        
        # Get boxes coordinates
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # Calculate areas
        area = torch.mul(x2 - x1, y2 - y1)
        
        count = 0
        while idx.numel() > 0:
            i = idx[0]
            keep[count] = i
            count += 1
            
            if idx.size(0) == 1:
                break
            
            idx = idx[1:]
            
            # Calculate IoU
            xx1 = x1[idx].clamp(min=float(x1[i]))
            yy1 = y1[idx].clamp(min=float(y1[i]))
            xx2 = x2[idx].clamp(max=float(x2[i]))
            yy2 = y2[idx].clamp(max=float(y2[i]))
            
            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h
            
            union = area[i] + area[idx] - inter
            IoU = inter / union
            
            # Keep boxes with IoU less than threshold
            idx = idx[IoU.le(overlap)]
        
        return keep, count