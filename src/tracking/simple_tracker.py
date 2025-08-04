import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from loguru import logger


class SimpleTracker:
    """Simple IoU-based object tracker for sam48 pipeline."""
    
    def __init__(self, 
                 track_thresh: float = 0.5,
                 match_thresh: float = 0.3,
                 max_disappeared: int = 30):
        """
        Initialize simple tracker.
        
        Args:
            track_thresh: Minimum confidence for starting new tracks
            match_thresh: IoU threshold for matching detections to tracks
            max_disappeared: Maximum frames a track can be missing before removal
        """
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_disappeared = max_disappeared
        
        self.next_track_id = 1
        self.tracks = {}  # track_id -> track_info
        self.disappeared = {}  # track_id -> frames_disappeared
        
        logger.info(f"SimpleTracker initialized with track_thresh={track_thresh}")
        
    def update(self, detections: List[Dict], frame_id: int) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries
            frame_id: Current frame ID
            
        Returns:
            List of tracking results with track IDs
        """
        # If no existing tracks, initialize new ones
        if len(self.tracks) == 0:
            for detection in detections:
                if detection['confidence'] >= self.track_thresh:
                    self._register_track(detection, frame_id)
        else:
            # Compute IoU matrix between existing tracks and detections
            track_ids = list(self.tracks.keys())
            track_boxes = [self.tracks[tid]['bbox'] for tid in track_ids]
            det_boxes = [det['bbox'] for det in detections]
            
            if len(track_boxes) > 0 and len(det_boxes) > 0:
                iou_matrix = self._compute_iou_matrix(track_boxes, det_boxes)
                
                # Perform matching using Hungarian algorithm (simplified greedy approach)
                matches, unmatched_tracks, unmatched_detections = self._associate_detections_to_tracks(
                    iou_matrix, self.match_thresh
                )
                
                # Update matched tracks
                for track_idx, det_idx in matches:
                    track_id = track_ids[track_idx]
                    detection = detections[det_idx]
                    
                    # Update track
                    self.tracks[track_id].update({
                        'bbox': detection['bbox'],
                        'class_name': detection['class_name'],
                        'confidence': detection['confidence'],
                        'frame_id': frame_id,
                        'center': self._get_bbox_center(detection['bbox'])
                    })
                    
                    # Reset disappeared counter
                    if track_id in self.disappeared:
                        del self.disappeared[track_id]
                        
                # Mark unmatched tracks as disappeared
                for track_idx in unmatched_tracks:
                    track_id = track_ids[track_idx]
                    if track_id not in self.disappeared:
                        self.disappeared[track_id] = 0
                    self.disappeared[track_id] += 1
                    
                # Create new tracks for unmatched detections
                for det_idx in unmatched_detections:
                    detection = detections[det_idx]
                    if detection['confidence'] >= self.track_thresh:
                        self._register_track(detection, frame_id)
                        
        # Remove tracks that have been missing for too long
        tracks_to_remove = []
        for track_id, frames_disappeared in self.disappeared.items():
            if frames_disappeared > self.max_disappeared:
                tracks_to_remove.append(track_id)
                
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            del self.disappeared[track_id]
            
        # Generate tracking results
        tracking_results = []
        for track_id, track_info in self.tracks.items():
            if track_id not in self.disappeared:  # Only return active tracks
                result = {
                    'frame_id': frame_id,
                    'track_id': track_id,
                    'bbox': track_info['bbox'],
                    'class_name': track_info['class_name'],
                    'confidence': track_info['confidence'],
                    'center': track_info['center']
                }
                tracking_results.append(result)
                
        logger.debug(f"Frame {frame_id}: {len(tracking_results)} active tracks")
        return tracking_results
        
    def _register_track(self, detection: Dict, frame_id: int):
        """Register a new track."""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        center = self._get_bbox_center(detection['bbox'])
        
        self.tracks[track_id] = {
            'bbox': detection['bbox'],
            'class_name': detection['class_name'],
            'confidence': detection['confidence'],
            'frame_id': frame_id,
            'start_frame': frame_id,
            'center': center,
            'history': [center]  # Track center history for motion analysis
        }
        
        logger.debug(f"Registered new track {track_id} at frame {frame_id}")
        
    def _get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return (cx, cy)
        
    def _compute_iou_matrix(self, boxes1: List[List[float]], boxes2: List[List[float]]) -> np.ndarray:
        """Compute IoU matrix between two sets of bounding boxes."""
        matrix = np.zeros((len(boxes1), len(boxes2)))
        
        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                matrix[i, j] = self._compute_iou(box1, box2)
                
        return matrix
        
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
            
        return intersection / union
        
    def _associate_detections_to_tracks(self, iou_matrix: np.ndarray, threshold: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections to tracks using greedy matching."""
        matches = []
        unmatched_tracks = list(range(iou_matrix.shape[0]))
        unmatched_detections = list(range(iou_matrix.shape[1]))
        
        # Find matches using greedy approach
        while True:
            if len(unmatched_tracks) == 0 or len(unmatched_detections) == 0:
                break
                
            # Find best match
            best_iou = 0
            best_track_idx = None
            best_det_idx = None
            
            for track_idx in unmatched_tracks:
                for det_idx in unmatched_detections:
                    iou = iou_matrix[track_idx, det_idx]
                    if iou > best_iou and iou >= threshold:
                        best_iou = iou
                        best_track_idx = track_idx
                        best_det_idx = det_idx
                        
            if best_track_idx is not None and best_det_idx is not None:
                matches.append((best_track_idx, best_det_idx))
                unmatched_tracks.remove(best_track_idx)
                unmatched_detections.remove(best_det_idx)
            else:
                break
                
        return matches, unmatched_tracks, unmatched_detections
        
    def get_track_info(self, track_id: int) -> Optional[Dict]:
        """Get information about a specific track."""
        return self.tracks.get(track_id)
        
    def get_active_tracks(self) -> Dict[int, Dict]:
        """Get all active tracks (not disappeared)."""
        active_tracks = {}
        for track_id, track_info in self.tracks.items():
            if track_id not in self.disappeared:
                active_tracks[track_id] = track_info
        return active_tracks