import json
import csv
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger


class ProcessingLogger:
    """Comprehensive logging system for sam48 pipeline results."""
    
    def __init__(self, 
                 output_dir: str = "output/logs",
                 log_format: str = "json",
                 session_id: Optional[str] = None):
        """
        Initialize processing logger.
        
        Args:
            output_dir: Directory to save log files
            log_format: Log format ("json" or "csv")
            session_id: Unique session identifier
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_format = log_format
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Data storage
        self.detection_log = []
        self.tracking_log = []
        self.roi_events_log = []
        self.sam_analysis_log = []
        self.classification_log = []
        
        # File paths
        self.detection_file = self.output_dir / f"detections_{self.session_id}.{log_format}"
        self.tracking_file = self.output_dir / f"tracking_{self.session_id}.{log_format}"
        self.roi_events_file = self.output_dir / f"roi_events_{self.session_id}.{log_format}"
        self.sam_analysis_file = self.output_dir / f"sam_analysis_{self.session_id}.{log_format}"
        self.classification_file = self.output_dir / f"classification_{self.session_id}.{log_format}"
        self.summary_file = self.output_dir / f"session_summary_{self.session_id}.json"
        
        # Session metadata
        self.session_start = datetime.now()
        self.session_metadata = {
            'session_id': self.session_id,
            'start_time': self.session_start.isoformat(),
            'log_format': log_format,
            'files': {
                'detections': str(self.detection_file),
                'tracking': str(self.tracking_file),
                'roi_events': str(self.roi_events_file),
                'sam_analysis': str(self.sam_analysis_file),
                'classification': str(self.classification_file),
                'summary': str(self.summary_file)
            }
        }
        
        logger.info(f"Processing logger initialized: session {self.session_id}")
        
    def log_detections(self, detections: List[Dict], frame_id: int):
        """Log object detection results."""
        timestamp = datetime.now().isoformat()
        
        for detection in detections:
            log_entry = {
                'timestamp': timestamp,
                'session_id': self.session_id,
                'frame_id': frame_id,
                'bbox': detection['bbox'],
                'class_name': detection['class_name'],
                'confidence': detection['confidence'],
                'class_id': detection.get('class_id')
            }
            self.detection_log.append(log_entry)
            
        logger.debug(f"Logged {len(detections)} detections for frame {frame_id}")
        
    def log_tracking(self, tracking_results: List[Dict], frame_id: int):
        """Log object tracking results."""
        timestamp = datetime.now().isoformat()
        
        for result in tracking_results:
            log_entry = {
                'timestamp': timestamp,
                'session_id': self.session_id,
                'frame_id': frame_id,
                'track_id': result['track_id'],
                'bbox': result['bbox'],
                'class_name': result['class_name'],
                'confidence': result['confidence'],
                'center': result.get('center')
            }
            self.tracking_log.append(log_entry)
            
        logger.debug(f"Logged {len(tracking_results)} tracks for frame {frame_id}")
        
    def log_roi_events(self, roi_events: Dict):
        """Log ROI monitoring events."""
        timestamp = datetime.now().isoformat()
        frame_id = roi_events['frame_id']
        
        # Log entry events
        for entry in roi_events.get('entries', []):
            log_entry = {
                'timestamp': timestamp,
                'session_id': self.session_id,
                'frame_id': frame_id,
                'event_type': 'entry',
                'track_id': entry['track_id'],
                'center': entry['center'],
                'class_name': entry['class_name']
            }
            self.roi_events_log.append(log_entry)
            
        # Log exit events
        for exit_event in roi_events.get('exits', []):
            log_entry = {
                'timestamp': timestamp,
                'session_id': self.session_id,
                'frame_id': frame_id,
                'event_type': 'exit',
                'track_id': exit_event['track_id'],
                'entry_frame': exit_event.get('entry_frame'),
                'frames_in_roi': exit_event['frames_in_roi'],
                'triggered': exit_event['triggered']
            }
            self.roi_events_log.append(log_entry)
            
        # Log trigger events
        for triggered in roi_events.get('triggered', []):
            log_entry = {
                'timestamp': timestamp,
                'session_id': self.session_id,
                'frame_id': frame_id,
                'event_type': 'trigger',
                'track_id': triggered['track_id'],
                'entry_frame': triggered['entry_frame'],
                'frames_in_roi': triggered['frames_in_roi'],
                'center': triggered['center'],
                'bbox': triggered['bbox'],
                'class_name': triggered['class_name']
            }
            self.roi_events_log.append(log_entry)
            
        total_events = len(roi_events.get('entries', [])) + len(roi_events.get('exits', [])) + len(roi_events.get('triggered', []))
        if total_events > 0:
            logger.debug(f"Logged {total_events} ROI events for frame {frame_id}")
            
    def log_sam_analysis(self, sam_result: Dict, frame_id: int):
        """Log SAM segmentation analysis results."""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'session_id': self.session_id,
            'frame_id': frame_id,
            'track_id': sam_result['track_id'],
            'success': sam_result['success'],
            'score': sam_result.get('score'),
            'bbox': sam_result.get('bbox'),
            'center_point': sam_result.get('center_point'),
            'mask_path': sam_result.get('mask_path'),
            'mask_stats': sam_result.get('stats'),
            'fallback': sam_result.get('fallback', False)
        }
        
        self.sam_analysis_log.append(log_entry)
        logger.debug(f"Logged SAM analysis for track {sam_result['track_id']} at frame {frame_id}")
        
    def log_classification(self, classification_result: Dict, frame_id: int, features: Dict = None):
        """Log object classification refinement results."""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'session_id': self.session_id,
            'frame_id': frame_id,
            'track_id': classification_result['track_id'],
            'initial_class': classification_result['initial_class'],
            'refined_class': classification_result['refined_class'],
            'confidence': classification_result['confidence'],
            'confidence_scores': classification_result['confidence_scores'],
            'changed': classification_result['changed'],
            'reasoning': classification_result['reasoning']
        }
        
        # Add feature summary if provided
        if features and 'combined_features' in features:
            combined = features['combined_features']
            log_entry['feature_dimensions'] = {
                'color_dim': combined.get('color_dim', 0),
                'shape_dim': combined.get('shape_dim', 0),
                'texture_dim': combined.get('texture_dim', 0),
                'full_dim': combined.get('full_dim', 0)
            }
            
        self.classification_log.append(log_entry)
        logger.debug(f"Logged classification for track {classification_result['track_id']} at frame {frame_id}")
        
    def save_logs(self):
        """Save all logs to files."""
        try:
            if self.log_format == "json":
                self._save_json_logs()
            else:
                self._save_csv_logs()
                
            # Always save session summary as JSON
            self._save_session_summary()
            
            logger.info(f"All logs saved for session {self.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to save logs: {e}")
            
    def _save_json_logs(self):
        """Save logs in JSON format."""
        if self.detection_log:
            with open(self.detection_file, 'w') as f:
                json.dump(self.detection_log, f, indent=2, default=str)
                
        if self.tracking_log:
            with open(self.tracking_file, 'w') as f:
                json.dump(self.tracking_log, f, indent=2, default=str)
                
        if self.roi_events_log:
            with open(self.roi_events_file, 'w') as f:
                json.dump(self.roi_events_log, f, indent=2, default=str)
                
        if self.sam_analysis_log:
            with open(self.sam_analysis_file, 'w') as f:
                json.dump(self.sam_analysis_log, f, indent=2, default=str)
                
        if self.classification_log:
            with open(self.classification_file, 'w') as f:
                json.dump(self.classification_log, f, indent=2, default=str)
                
    def _save_csv_logs(self):
        """Save logs in CSV format."""
        if self.detection_log:
            df = pd.DataFrame(self.detection_log)
            df.to_csv(self.detection_file, index=False)
            
        if self.tracking_log:
            df = pd.DataFrame(self.tracking_log)
            df.to_csv(self.tracking_file, index=False)
            
        if self.roi_events_log:
            df = pd.DataFrame(self.roi_events_log)
            df.to_csv(self.roi_events_file, index=False)
            
        if self.sam_analysis_log:
            df = pd.DataFrame(self.sam_analysis_log)
            df.to_csv(self.sam_analysis_file, index=False)
            
        if self.classification_log:
            df = pd.DataFrame(self.classification_log)
            df.to_csv(self.classification_file, index=False)
            
    def _save_session_summary(self):
        """Save session summary with statistics."""
        session_end = datetime.now()
        
        summary = {
            **self.session_metadata,
            'end_time': session_end.isoformat(),
            'duration_seconds': (session_end - self.session_start).total_seconds(),
            'statistics': {
                'total_detections': len(self.detection_log),
                'total_tracks': len(set(entry['track_id'] for entry in self.tracking_log)),
                'roi_entries': len([e for e in self.roi_events_log if e['event_type'] == 'entry']),
                'roi_exits': len([e for e in self.roi_events_log if e['event_type'] == 'exit']),
                'sam_triggers': len([e for e in self.roi_events_log if e['event_type'] == 'trigger']),
                'sam_analyses': len(self.sam_analysis_log),
                'classifications': len(self.classification_log),
                'classification_changes': len([c for c in self.classification_log if c['changed']])
            }
        }
        
        # Add per-class statistics
        if self.classification_log:
            class_stats = {}
            for entry in self.classification_log:
                initial = entry['initial_class']
                refined = entry['refined_class']
                
                if initial not in class_stats:
                    class_stats[initial] = {'initial_count': 0, 'refined_to': {}}
                class_stats[initial]['initial_count'] += 1
                
                if refined not in class_stats[initial]['refined_to']:
                    class_stats[initial]['refined_to'][refined] = 0
                class_stats[initial]['refined_to'][refined] += 1
                
            summary['class_statistics'] = class_stats
            
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
    def get_statistics(self) -> Dict:
        """Get current session statistics."""
        return {
            'session_id': self.session_id,
            'detections': len(self.detection_log),
            'tracks': len(set(entry['track_id'] for entry in self.tracking_log)),
            'roi_events': len(self.roi_events_log),
            'sam_analyses': len(self.sam_analysis_log),
            'classifications': len(self.classification_log)
        }
        
    def export_track_timeline(self, track_id: int, output_path: Optional[str] = None) -> Dict:
        """Export complete timeline for a specific track."""
        timeline = {
            'track_id': track_id,
            'timeline': []
        }
        
        # Collect all events for this track
        events = []
        
        # Add tracking events
        for entry in self.tracking_log:
            if entry['track_id'] == track_id:
                events.append({
                    'frame_id': entry['frame_id'],
                    'timestamp': entry['timestamp'],
                    'event_type': 'tracking',
                    'data': entry
                })
                
        # Add ROI events
        for entry in self.roi_events_log:
            if entry['track_id'] == track_id:
                events.append({
                    'frame_id': entry['frame_id'],
                    'timestamp': entry['timestamp'],
                    'event_type': f"roi_{entry['event_type']}",
                    'data': entry
                })
                
        # Add SAM analysis events
        for entry in self.sam_analysis_log:
            if entry['track_id'] == track_id:
                events.append({
                    'frame_id': entry['frame_id'],
                    'timestamp': entry['timestamp'],
                    'event_type': 'sam_analysis',
                    'data': entry
                })
                
        # Add classification events
        for entry in self.classification_log:
            if entry['track_id'] == track_id:
                events.append({
                    'frame_id': entry['frame_id'],
                    'timestamp': entry['timestamp'],
                    'event_type': 'classification',
                    'data': entry
                })
                
        # Sort by frame_id
        events.sort(key=lambda x: x['frame_id'])
        timeline['timeline'] = events
        
        # Save if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(timeline, f, indent=2, default=str)
                
        return timeline