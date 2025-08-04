import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from loguru import logger


class DebugVisualizer:
    """Debug video generation with all processing stages overlay."""
    
    def __init__(self, 
                 frame_size: Tuple[int, int] = (1024, 1024),
                 info_panel_height: int = 200,
                 colors: Optional[Dict[str, Tuple[int, int, int]]] = None):
        """
        Initialize debug visualizer.
        
        Args:
            frame_size: Video frame size (width, height)
            info_panel_height: Height of info panel in pixels
            colors: Color scheme for different elements
        """
        self.frame_size = frame_size
        self.info_panel_height = info_panel_height
        self.output_size = (frame_size[0], frame_size[1] + info_panel_height)
        
        # Default color scheme
        self.colors = colors or {
            'bbox': (0, 255, 0),           # Green for bounding boxes
            'track_id': (255, 255, 0),     # Cyan for track IDs
            'roi': (0, 255, 255),          # Yellow for ROI
            'entry': (0, 255, 0),          # Green for entry events
            'exit': (0, 0, 255),           # Red for exit events
            'sam_trigger': (0, 165, 255),  # Orange for SAM triggers
            'sam_mask': (255, 0, 255),     # Magenta for SAM masks
            'classification': (255, 255, 255), # White for classification text
            'info_bg': (40, 40, 40),       # Dark gray for info panel
            'info_text': (255, 255, 255),  # White for info text
            'status_active': (0, 255, 0),  # Green for active status
            'status_triggered': (0, 165, 255), # Orange for triggered status
        }
        
        logger.info(f"Debug visualizer initialized: {frame_size} + {info_panel_height}px panel")
        
    def create_debug_frame(self,
                          original_frame: np.ndarray,
                          detections: List[Dict],
                          tracking_results: List[Dict],
                          roi_events: Dict,
                          sam_results: Dict,
                          classification_results: Dict,
                          frame_id: int,
                          roi_monitor=None,
                          pipeline_stats: Optional[Dict] = None) -> np.ndarray:
        """
        Create comprehensive debug frame with all processing stages.
        
        Args:
            original_frame: Original video frame
            detections: YOLO detection results
            tracking_results: Tracking results
            roi_events: ROI monitoring events
            sam_results: SAM analysis results (track_id -> result)
            classification_results: Classification results (track_id -> result)
            frame_id: Current frame ID
            roi_monitor: ROI monitor instance for visualization
            pipeline_stats: Pipeline statistics
            
        Returns:
            Debug frame with all overlays and info panel
        """
        # Create output frame with info panel
        debug_frame = np.zeros((self.output_size[1], self.output_size[0], 3), dtype=np.uint8)
        
        # Copy original frame to top portion
        debug_frame[:self.frame_size[1], :self.frame_size[0]] = original_frame.copy()
        
        # Draw all processing stages
        self._draw_detections(debug_frame, detections)
        self._draw_tracking(debug_frame, tracking_results)
        self._draw_roi(debug_frame, roi_events, roi_monitor)
        self._draw_sam_results(debug_frame, sam_results)
        self._draw_classification_results(debug_frame, classification_results)
        
        # Draw info panel
        self._draw_info_panel(debug_frame, frame_id, tracking_results, roi_events, 
                             sam_results, classification_results, pipeline_stats)
        
        return debug_frame
        
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict]):
        """Draw detection bounding boxes."""
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['bbox'], 2)
            
            # Draw detection label
            label = f"{class_name} {confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Background for text
            cv2.rectangle(frame, (x1, y1 - text_height - baseline - 5),
                         (x1 + text_width, y1), self.colors['bbox'], -1)
            
            # Text
            cv2.putText(frame, label, (x1, y1 - baseline - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                       
    def _draw_tracking(self, frame: np.ndarray, tracking_results: List[Dict]):
        """Draw tracking information."""
        for result in tracking_results:
            bbox = result['bbox']
            track_id = result['track_id']
            center = result.get('center')
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Calculate center if not provided
            if center is None:
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # Draw track ID
            track_label = f"ID:{track_id}"
            cv2.putText(frame, track_label, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['track_id'], 2)
            
            # Draw center point
            cv2.circle(frame, (int(center[0]), int(center[1])), 4, self.colors['track_id'], -1)
            
    def _draw_roi(self, frame: np.ndarray, roi_events: Dict, roi_monitor=None):
        """Draw ROI and related events."""
        if roi_monitor is not None:
            # Draw ROI polygon
            polygon = roi_monitor.get_roi_polygon()
            cv2.polylines(frame, [polygon], isClosed=True, 
                         color=self.colors['roi'], thickness=2)
            
            # Add ROI label
            if len(polygon) > 0:
                first_point = polygon[0]
                cv2.putText(frame, "ROI", (first_point[0], first_point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['roi'], 2)
        
        # Draw entry events
        for entry in roi_events.get('entries', []):
            center = entry['center']
            cv2.circle(frame, (int(center[0]), int(center[1])), 12, self.colors['entry'], 3)
            cv2.putText(frame, "ENTRY", (int(center[0]) - 25, int(center[1]) - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['entry'], 2)
        
        # Draw SAM trigger events
        for triggered in roi_events.get('triggered', []):
            center = triggered['center']
            cv2.circle(frame, (int(center[0]), int(center[1])), 18, self.colors['sam_trigger'], 4)
            cv2.putText(frame, "SAM TRIGGERED", (int(center[0]) - 45, int(center[1]) - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['sam_trigger'], 2)
        
        # Show objects in ROI with frame counts
        for obj_info in roi_events.get('in_roi', []):
            center = obj_info['center']
            frames_count = obj_info['frames_in_roi']
            triggered = obj_info.get('triggered', False)
            
            color = self.colors['sam_trigger'] if triggered else self.colors['roi']
            
            cv2.putText(frame, f"F:{frames_count}", 
                       (int(center[0]) - 15, int(center[1]) + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                       
    def _draw_sam_results(self, frame: np.ndarray, sam_results: Dict):
        """Draw SAM segmentation results."""
        for track_id, result in sam_results.items():
            if result.get('success') and 'mask' in result:
                mask = result['mask']
                
                # Create colored overlay for mask
                overlay = np.zeros_like(frame)
                overlay[mask] = self.colors['sam_mask']
                
                # Blend with original frame
                alpha = 0.3
                mask_area = frame[:self.frame_size[1], :self.frame_size[0]]
                mask_area[mask] = cv2.addWeighted(
                    mask_area[mask], 1 - alpha, 
                    overlay[:self.frame_size[1], :self.frame_size[0]][mask], alpha, 0
                )
                
                # Add SAM label
                if 'center_point' in result:
                    center = result['center_point']
                    cv2.putText(frame, f"SAM:{track_id}", 
                               (int(center[0]) - 30, int(center[1]) + 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['sam_mask'], 2)
                               
    def _draw_classification_results(self, frame: np.ndarray, classification_results: Dict):
        """Draw classification refinement results."""
        for track_id, result in classification_results.items():
            if result.get('changed'):
                # Find corresponding track position (would need to be passed or stored)
                # For now, just indicate classification changes occurred
                pass
                
    def _draw_info_panel(self, 
                        frame: np.ndarray, 
                        frame_id: int,
                        tracking_results: List[Dict],
                        roi_events: Dict,
                        sam_results: Dict,
                        classification_results: Dict,
                        pipeline_stats: Optional[Dict]):
        """Draw comprehensive info panel."""
        panel_y_start = self.frame_size[1]
        
        # Fill info panel background
        cv2.rectangle(frame, (0, panel_y_start), 
                     (self.frame_size[0], self.output_size[1]), 
                     self.colors['info_bg'], -1)
        
        # Panel layout
        col1_x = 20
        col2_x = 300
        col3_x = 600
        col4_x = 850
        text_y_start = panel_y_start + 25
        line_height = 22
        
        # Column 1: Frame & Time Info
        current_time = datetime.now().strftime("%H:%M:%S")
        info_lines_1 = [
            f"Frame: {frame_id}",
            f"Time: {current_time}",
            f"Active Tracks: {len(tracking_results)}",
            f"In ROI: {len(roi_events.get('in_roi', []))}",
        ]
        
        for i, line in enumerate(info_lines_1):
            y_pos = text_y_start + i * line_height
            cv2.putText(frame, line, (col1_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['info_text'], 1)
        
        # Column 2: ROI Events
        info_lines_2 = [
            "ROI EVENTS:",
            f"Entries: {len(roi_events.get('entries', []))}",
            f"Exits: {len(roi_events.get('exits', []))}",
            f"SAM Triggers: {len(roi_events.get('triggered', []))}",
        ]
        
        for i, line in enumerate(info_lines_2):
            y_pos = text_y_start + i * line_height
            color = self.colors['info_text'] if i == 0 else self.colors['status_active']
            cv2.putText(frame, line, (col2_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Column 3: Processing Status
        sam_count = len([r for r in sam_results.values() if r.get('success')])
        classification_changes = len([r for r in classification_results.values() if r.get('changed')])
        
        info_lines_3 = [
            "PROCESSING:",
            f"SAM Analyses: {sam_count}",
            f"Classifications: {classification_changes}",
            f"Pipeline: {'RUNNING' if pipeline_stats else 'IDLE'}",
        ]
        
        for i, line in enumerate(info_lines_3):
            y_pos = text_y_start + i * line_height
            color = self.colors['info_text'] if i == 0 else self.colors['status_active']
            cv2.putText(frame, line, (col3_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Column 4: Active Objects Detail
        cv2.putText(frame, "ACTIVE OBJECTS:", (col4_x, text_y_start),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['info_text'], 1)
        
        # Show details for up to 7 active objects
        y_offset = text_y_start + line_height
        max_objects = min(7, len(tracking_results))
        
        for i, result in enumerate(tracking_results[:max_objects]):
            track_id = result['track_id']
            class_name = result['class_name']
            
            # Check if object is in ROI
            in_roi = any(obj['track_id'] == track_id for obj in roi_events.get('in_roi', []))
            triggered = track_id in sam_results
            
            # Status indicator
            if triggered:
                status = "SAM"
                color = self.colors['sam_trigger']
            elif in_roi:
                status = "ROI"
                color = self.colors['status_triggered']
            else:
                status = "TRK"
                color = self.colors['status_active']
            
            object_info = f"ID{track_id}:{class_name}[{status}]"
            cv2.putText(frame, object_info, (col4_x, y_offset + i * (line_height - 2)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Bottom status line
        bottom_y = self.output_size[1] - 10
        if pipeline_stats:
            progress = pipeline_stats.get('progress', 0)
            status_line = f"Progress: {progress*100:.1f}% | Total Objects: {pipeline_stats.get('active_tracks', 0)}"
            cv2.putText(frame, status_line, (col1_x, bottom_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['info_text'], 1)
        
        # Draw separator line
        cv2.line(frame, (0, panel_y_start), (self.frame_size[0], panel_y_start), 
                (100, 100, 100), 2)
                
    def add_legend(self, frame: np.ndarray) -> np.ndarray:
        """Add color legend to frame."""
        legend_items = [
            ("Detection", self.colors['bbox']),
            ("Track ID", self.colors['track_id']),
            ("ROI", self.colors['roi']),
            ("Entry", self.colors['entry']),
            ("SAM Trigger", self.colors['sam_trigger']),
            ("SAM Mask", self.colors['sam_mask']),
        ]
        
        # Legend position (top-right corner)
        legend_x = self.frame_size[0] - 150
        legend_y = 20
        
        # Background for legend
        legend_height = len(legend_items) * 20 + 20
        cv2.rectangle(frame, (legend_x - 10, legend_y - 10),
                     (self.frame_size[0] - 10, legend_y + legend_height),
                     (0, 0, 0), -1)  # Black background
        
        for i, (label, color) in enumerate(legend_items):
            y_pos = legend_y + i * 20
            
            # Color square
            cv2.rectangle(frame, (legend_x, y_pos), (legend_x + 15, y_pos + 10), color, -1)
            
            # Label
            cv2.putText(frame, label, (legend_x + 20, y_pos + 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
        
    def create_summary_frame(self, pipeline_stats: Dict) -> np.ndarray:
        """Create summary frame with final statistics."""
        summary_frame = np.zeros((self.output_size[1], self.output_size[0], 3), dtype=np.uint8)
        
        # Title
        title = "SAM48 PROCESSING COMPLETE"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]
        title_x = (self.frame_size[0] - title_size[0]) // 2
        cv2.putText(summary_frame, title, (title_x, 100),
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)
        
        # Statistics
        stats_y = 200
        line_height = 40
        
        stats_lines = [
            f"Total Frames: {pipeline_stats.get('total_frames', 0)}",
            f"Objects Detected: {pipeline_stats.get('detections', 0)}",
            f"Unique Tracks: {pipeline_stats.get('tracks', 0)}",
            f"ROI Events: {pipeline_stats.get('roi_events', 0)}",
            f"SAM Analyses: {pipeline_stats.get('sam_analyses', 0)}",
            f"Classifications: {pipeline_stats.get('classifications', 0)}",
        ]
        
        for i, line in enumerate(stats_lines):
            y_pos = stats_y + i * line_height
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (self.frame_size[0] - text_size[0]) // 2
            cv2.putText(summary_frame, line, (text_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return summary_frame