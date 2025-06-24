"""Face detection and tracking utilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from syncnet.core.types import BBox, Detection, Track


@dataclass
class FaceTrack:
    """Represents a face track across multiple frames."""
    
    track_id: int
    detections: list[Detection] = field(default_factory=list)
    last_seen_frame: int = -1
    failed_detections: int = 0
    
    @property
    def bboxes(self) -> np.ndarray:
        """Get all bounding boxes as numpy array."""
        return np.array([d["bbox"] for d in self.detections])
    
    @property
    def start_frame(self) -> int:
        """Get first frame of track."""
        return self.detections[0]["frame_idx"] if self.detections else -1
    
    @property
    def end_frame(self) -> int:
        """Get last frame of track."""
        return self.detections[-1]["frame_idx"] if self.detections else -1
    
    @property
    def length(self) -> int:
        """Get track length."""
        return len(self.detections)
    
    @property
    def mean_confidence(self) -> float:
        """Get mean detection confidence."""
        if not self.detections:
            return 0.0
        return np.mean([d["confidence"] for d in self.detections])
    
    @property
    def mean_size(self) -> float:
        """Get mean face size."""
        if not self.detections:
            return 0.0
        bboxes = self.bboxes
        widths = bboxes[:, 2] - bboxes[:, 0]
        heights = bboxes[:, 3] - bboxes[:, 1]
        return float(np.mean(np.maximum(widths, heights)))


class FaceTracker:
    """Track faces across video frames using IoU matching."""
    
    def __init__(
        self,
        min_track_length: int = 50,
        min_face_size: int = 100,
        max_failed_detections: int = 25,
        iou_threshold: float = 0.5
    ) -> None:
        self.min_track_length = min_track_length
        self.min_face_size = min_face_size
        self.max_failed_detections = max_failed_detections
        self.iou_threshold = iou_threshold
        
        self.tracks: dict[int, FaceTrack] = {}
        self.next_track_id = 0
        self.finished_tracks: list[FaceTrack] = []
    
    def update(self, detections: list[Detection], frame_idx: int) -> None:
        """Update tracks with new detections.
        
        Args:
            detections: Face detections for current frame
            frame_idx: Current frame index
        """
        # Update frame index for all detections
        for det in detections:
            det["frame_idx"] = frame_idx
        
        # Match detections to existing tracks
        matched_tracks = set()
        matched_detections = set()
        
        for det_idx, detection in enumerate(detections):
            best_track_id = None
            best_iou = self.iou_threshold
            
            # Find best matching track
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                
                # Get last detection from track
                if not track.detections:
                    continue
                
                last_det = track.detections[-1]
                iou = self._calculate_iou(detection["bbox"], last_det["bbox"])
                
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            
            # Assign detection to track
            if best_track_id is not None:
                self.tracks[best_track_id].detections.append(detection)
                self.tracks[best_track_id].last_seen_frame = frame_idx
                self.tracks[best_track_id].failed_detections = 0
                matched_tracks.add(best_track_id)
                matched_detections.add(det_idx)
        
        # Create new tracks for unmatched detections
        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detections:
                track = FaceTrack(track_id=self.next_track_id)
                track.detections.append(detection)
                track.last_seen_frame = frame_idx
                self.tracks[self.next_track_id] = track
                self.next_track_id += 1
        
        # Update failed detection counts
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track_id not in matched_tracks:
                track.failed_detections += 1
                
                # Remove track if too many failed detections
                if track.failed_detections > self.max_failed_detections:
                    tracks_to_remove.append(track_id)
        
        # Move finished tracks
        for track_id in tracks_to_remove:
            track = self.tracks.pop(track_id)
            if self._is_valid_track(track):
                self.finished_tracks.append(track)
    
    def get_tracks(self) -> list[Track]:
        """Get all valid tracks.
        
        Returns:
            List of tracks meeting minimum requirements
        """
        # Finish all remaining tracks
        all_tracks = list(self.finished_tracks)
        
        for track in self.tracks.values():
            if self._is_valid_track(track):
                all_tracks.append(track)
        
        # Convert to Track format
        result = []
        for track in all_tracks:
            if track.length > 0:
                # Get frame indices
                frame_indices = [d["frame_idx"] for d in track.detections]
                
                # Create dense bbox array
                min_frame = min(frame_indices)
                max_frame = max(frame_indices)
                num_frames = max_frame - min_frame + 1
                
                bbox_array = np.zeros((num_frames, 4))
                for det in track.detections:
                    idx = det["frame_idx"] - min_frame
                    bbox_array[idx] = det["bbox"]
                
                # Interpolate missing frames
                bbox_array = self._interpolate_bboxes(bbox_array, frame_indices, min_frame)
                
                result.append(Track(
                    frame=bbox_array,
                    bbox=bbox_array,
                    start_frame=min_frame,
                    end_frame=max_frame
                ))
        
        return result
    
    def _is_valid_track(self, track: FaceTrack) -> bool:
        """Check if track meets minimum requirements."""
        return (
            track.length >= self.min_track_length and
            track.mean_size >= self.min_face_size
        )
    
    def _calculate_iou(self, bbox1: BBox, bbox2: BBox) -> float:
        """Calculate IoU between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-8)
    
    def _interpolate_bboxes(
        self,
        bbox_array: np.ndarray,
        frame_indices: list[int],
        min_frame: int
    ) -> np.ndarray:
        """Interpolate missing bounding boxes."""
        # Find missing frames
        all_frames = set(range(bbox_array.shape[0]))
        existing_frames = {idx - min_frame for idx in frame_indices}
        missing_frames = all_frames - existing_frames
        
        if not missing_frames:
            return bbox_array
        
        # Interpolate each coordinate
        for coord in range(4):
            valid_indices = [idx - min_frame for idx in frame_indices]
            valid_values = bbox_array[valid_indices, coord]
            
            # Linear interpolation
            for missing_idx in missing_frames:
                # Find nearest neighbors
                before_idx = max([idx for idx in valid_indices if idx < missing_idx], default=None)
                after_idx = min([idx for idx in valid_indices if idx > missing_idx], default=None)
                
                if before_idx is not None and after_idx is not None:
                    # Interpolate between neighbors
                    alpha = (missing_idx - before_idx) / (after_idx - before_idx)
                    bbox_array[missing_idx, coord] = (
                        (1 - alpha) * bbox_array[before_idx, coord] +
                        alpha * bbox_array[after_idx, coord]
                    )
                elif before_idx is not None:
                    # Use last known value
                    bbox_array[missing_idx, coord] = bbox_array[before_idx, coord]
                elif after_idx is not None:
                    # Use next known value
                    bbox_array[missing_idx, coord] = bbox_array[after_idx, coord]
        
        return bbox_array