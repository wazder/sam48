import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from loguru import logger


class ROIMonitor:
    """Monitors objects entering and staying within a Region of Interest defined by polygon."""
    
    def __init__(self, polygon_points: List[List[int]], trigger_frames: int = 10):
        """
        Initialize ROI monitor.
        
        Args:
            polygon_points: ROI polygon points [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            trigger_frames: Number of frames an object must stay in ROI to trigger event
        """
        self.polygon_points = np.array(polygon_points, dtype=np.int32)
        self.trigger_frames = trigger_frames
        
        # Track objects in ROI
        self.objects_in_roi = {}  # track_id -> frame_count
        self.entry_events = {}    # track_id -> entry_frame
        self.triggered_objects = set()  # track_ids that have triggered SAM analysis
        
        logger.info(f"ROI Monitor initialized: {len(polygon_points)} points, trigger={trigger_frames} frames")
        
    def update(self, tracking_results: List[Dict], frame_id: int) -> Dict:
        """
        Update ROI monitoring with new tracking results.
        
        Args:
            tracking_results: List of tracking result dictionaries
            frame_id: Current frame ID
            
        Returns:
            Dictionary with ROI events and status
        """
        current_in_roi = set()
        roi_events = {
            'entries': [],      # Objects that just entered ROI
            'exits': [],        # Objects that just exited ROI  
            'triggered': [],    # Objects that triggered SAM analysis
            'in_roi': [],       # All objects currently in ROI
            'frame_id': frame_id
        }
        
        # Check each tracked object
        for result in tracking_results:
            track_id = result['track_id']
            center = result.get('center')
            
            if center is None:
                # Calculate center if not provided
                bbox = result['bbox']
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            # Check if object center is inside polygon ROI
            if self._point_in_polygon(center):
                current_in_roi.add(track_id)
                
                # Track entry event
                if track_id not in self.objects_in_roi:
                    self.objects_in_roi[track_id] = 1
                    self.entry_events[track_id] = frame_id
                    roi_events['entries'].append({
                        'track_id': track_id,
                        'frame_id': frame_id,
                        'center': center,
                        'class_name': result['class_name']
                    })
                    logger.debug(f"Track {track_id} entered ROI at frame {frame_id}")
                else:
                    # Increment frame count for objects already in ROI
                    self.objects_in_roi[track_id] += 1
                    
                # Check if object should trigger SAM analysis
                if (self.objects_in_roi[track_id] >= self.trigger_frames and 
                    track_id not in self.triggered_objects):
                    
                    self.triggered_objects.add(track_id)
                    roi_events['triggered'].append({
                        'track_id': track_id,
                        'frame_id': frame_id,
                        'entry_frame': self.entry_events[track_id],
                        'frames_in_roi': self.objects_in_roi[track_id],
                        'center': center,
                        'bbox': result['bbox'],
                        'class_name': result['class_name']
                    })
                    logger.info(f"Track {track_id} triggered SAM analysis after {self.objects_in_roi[track_id]} frames in ROI")
                
                # Add to current in-ROI list
                roi_events['in_roi'].append({
                    'track_id': track_id,
                    'frames_in_roi': self.objects_in_roi[track_id],
                    'center': center,
                    'class_name': result['class_name'],
                    'triggered': track_id in self.triggered_objects
                })
        
        # Check for objects that exited ROI
        exited_tracks = set(self.objects_in_roi.keys()) - current_in_roi
        for track_id in exited_tracks:
            roi_events['exits'].append({
                'track_id': track_id,
                'frame_id': frame_id,
                'entry_frame': self.entry_events.get(track_id),
                'frames_in_roi': self.objects_in_roi[track_id],
                'triggered': track_id in self.triggered_objects
            })
            logger.debug(f"Track {track_id} exited ROI at frame {frame_id} after {self.objects_in_roi[track_id]} frames")
            
            # Clean up tracking data
            del self.objects_in_roi[track_id]
            if track_id in self.entry_events:
                del self.entry_events[track_id]
                
        return roi_events
        
    def _point_in_polygon(self, point: Tuple[float, float]) -> bool:
        """
        Check if a point is inside the ROI polygon using ray casting algorithm.
        
        Args:
            point: (x, y) coordinates
            
        Returns:
            True if point is inside polygon
        """
        x, y = point
        n = len(self.polygon_points)
        inside = False
        
        p1x, p1y = self.polygon_points[0]
        for i in range(1, n + 1):
            p2x, p2y = self.polygon_points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
        
    def get_roi_polygon(self) -> np.ndarray:
        """Get ROI polygon points for visualization."""
        return self.polygon_points.copy()
        
    def visualize_roi(self, frame: np.ndarray, 
                     roi_events: Dict = None,
                     color: Tuple[int, int, int] = (0, 255, 255),
                     thickness: int = 2) -> np.ndarray:
        """
        Draw ROI rectangle and events on frame.
        
        Args:
            frame: Input frame
            roi_events: ROI events from update()
            color: ROI rectangle color (BGR)
            thickness: Line thickness
            
        Returns:
            Frame with ROI visualization
        """
        frame_vis = frame.copy()
        
        # Draw ROI polygon
        cv2.polylines(frame_vis, [self.polygon_points], isClosed=True, color=color, thickness=thickness)
        
        # Add ROI label at first point
        first_point = self.polygon_points[0]
        cv2.putText(frame_vis, "ROI", (first_point[0], first_point[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if roi_events:
            # Visualize entry events (green circles)
            for entry in roi_events.get('entries', []):
                center = entry['center']
                cv2.circle(frame_vis, (int(center[0]), int(center[1])), 10, (0, 255, 0), -1)
                cv2.putText(frame_vis, "ENTRY", 
                           (int(center[0]) - 20, int(center[1]) - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Visualize exit events (red circles)  
            for exit_event in roi_events.get('exits', []):
                # Note: We don't have current center for exited objects
                # Could store last known position if needed
                pass
            
            # Visualize triggered events (orange circles)
            for triggered in roi_events.get('triggered', []):
                center = triggered['center']
                cv2.circle(frame_vis, (int(center[0]), int(center[1])), 15, (0, 165, 255), 3)
                cv2.putText(frame_vis, "SAM TRIGGERED", 
                           (int(center[0]) - 40, int(center[1]) - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            
            # Show objects currently in ROI with frame counts
            for obj_info in roi_events.get('in_roi', []):
                center = obj_info['center']
                frames_count = obj_info['frames_in_roi']
                triggered = obj_info['triggered']
                
                # Color based on status
                if triggered:
                    text_color = (0, 165, 255)  # Orange for triggered
                else:
                    text_color = (255, 255, 0)   # Cyan for in ROI
                
                cv2.putText(frame_vis, f"ID:{obj_info['track_id']} F:{frames_count}", 
                           (int(center[0]) - 30, int(center[1]) + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        
        return frame_vis
        
    def get_roi_stats(self) -> Dict:
        """Get current ROI statistics."""
        return {
            'objects_in_roi': len(self.objects_in_roi),
            'triggered_objects': len(self.triggered_objects),
            'polygon_points': self.polygon_points.tolist(),
            'trigger_frames': self.trigger_frames,
            'current_objects': dict(self.objects_in_roi)
        }
        
    def reset(self):
        """Reset ROI monitoring state."""
        self.objects_in_roi.clear()
        self.entry_events.clear()
        self.triggered_objects.clear()
        logger.info("ROI Monitor reset")
        
    def is_triggered(self, track_id: int) -> bool:
        """Check if a track has triggered SAM analysis."""
        return track_id in self.triggered_objects
        
    def get_frames_in_roi(self, track_id: int) -> int:
        """Get number of frames a track has been in ROI."""
        return self.objects_in_roi.get(track_id, 0)