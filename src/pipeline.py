import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm
from loguru import logger

from .utils import VideoHandler, VideoWriter, ROIMonitor, ProcessingLogger, DebugVisualizer
from .detection import YOLODetector
from .tracking import SimpleTracker
from .segmentation import SAMSegmenter
from .features import FeatureExtractor, ObjectClassifier


class Sam48Pipeline:
    """Main processing pipeline orchestrator for sam48."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize sam48 pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.video_handler = None
        self.detector = None
        self.tracker = None
        self.roi_monitor = None
        self.sam_segmenter = None
        self.feature_extractor = None
        self.classifier = None
        self.logger = None
        self.debug_visualizer = None
        self.debug_video_writer = None
        
        # Processing state
        self.current_frame_id = 0
        self.total_frames = 0
        
        logger.info("Sam48Pipeline initialized")
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
            
    def initialize_components(self) -> bool:
        """Initialize all pipeline components."""
        try:
            # Initialize YOLO detector
            detection_config = self.config['detection']
            self.detector = YOLODetector(
                model_path=detection_config['model'],
                target_classes=detection_config['classes'],
                confidence_threshold=detection_config['confidence_threshold'],
                iou_threshold=detection_config['iou_threshold']
            )
            
            # Initialize tracker
            tracking_config = self.config['tracking']
            self.tracker = SimpleTracker(
                track_thresh=tracking_config['track_thresh'],
                match_thresh=tracking_config['match_thresh'],
                max_disappeared=tracking_config['track_buffer']
            )
            
            # Initialize ROI monitor
            roi_config = self.config['roi']
            self.roi_monitor = ROIMonitor(
                polygon_points=roi_config['polygon_points'],
                trigger_frames=self.config['sam']['threshold']
            )
            
            # Initialize SAM segmenter
            sam_config = self.config['sam']
            self.sam_segmenter = SAMSegmenter(
                model_type=sam_config['model_type'],
                checkpoint_path=sam_config['checkpoint_path']
            )
            
            # Initialize feature extractor
            features_config = self.config['features']
            self.feature_extractor = FeatureExtractor(
                color_bins=features_config['color_bins']
            )
            
            # Initialize classifier
            self.classifier = ObjectClassifier()
            
            # Initialize logger
            logging_config = self.config['logging']
            self.logger = ProcessingLogger(
                output_dir=self.config['video']['output_dir'] + "/logs",
                log_format=logging_config['format']
            )
            
            # Initialize debug visualizer if enabled
            if self.config['debug']['debug_video']:
                target_resolution = tuple(self.config['video']['input_resolution'])
                self.debug_visualizer = DebugVisualizer(frame_size=target_resolution)
            
            logger.info("All pipeline components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False
            
    def process_video(self, video_path: str, max_frames: int = None) -> bool:
        """
        Process entire video through the pipeline.
        
        Args:
            video_path: Path to input video file
            max_frames: Maximum number of frames to process (None for all frames)
            
        Returns:
            True if processing successful
        """
        if not self.initialize_components():
            return False
            
        try:
            # Initialize video handler
            target_resolution = tuple(self.config['video']['input_resolution'])
            self.video_handler = VideoHandler(video_path, target_resolution)
            
            with self.video_handler:
                if not self.video_handler.is_opened:
                    logger.error("Failed to open video")
                    return False
                    
                self.total_frames = self.video_handler.total_frames
                
                # Apply frame limit if specified
                if max_frames is not None and max_frames < self.total_frames:
                    self.total_frames = max_frames
                    logger.info(f"Processing first {self.total_frames} frames from {video_path} (limited from {self.video_handler.total_frames})")
                else:
                    logger.info(f"Processing {self.total_frames} frames from {video_path}")
                
                # Initialize debug video writer if enabled
                if self.debug_visualizer:
                    debug_video_path = Path(self.config['video']['output_dir']) / "videos" / f"debug_{Path(video_path).stem}.mp4"
                    debug_video_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    self.debug_video_writer = VideoWriter(
                        str(debug_video_path),
                        self.debug_visualizer.output_size,
                        self.config['video']['fps']
                    )
                    self.debug_video_writer.open()
                    logger.info(f"Debug video will be saved to: {debug_video_path}")
                
                # Processing state for debug video
                sam_results = {}  # track_id -> sam_result
                classification_results = {}  # track_id -> classification_result
                
                # Process frames  
                frames_to_process = self.total_frames if max_frames is None else min(max_frames, self.total_frames)
                progress_bar = tqdm(
                    self.video_handler.frames(), 
                    total=frames_to_process,
                    desc="Processing frames"
                )
                
                frame_count = 0
                for frame_id, frame in progress_bar:
                    # Stop if we've reached the frame limit
                    if max_frames is not None and frame_count >= max_frames:
                        logger.info(f"Reached frame limit: {max_frames} (frame_id: {frame_id})")
                        break
                    
                    frame_count += 1
                        
                    self.current_frame_id = frame_id
                    
                    # Process single frame
                    detections, tracking_results, roi_events = self._process_frame(frame, frame_id, sam_results, classification_results)
                    
                    # Generate debug frame if enabled
                    if self.debug_visualizer and self.debug_video_writer:
                        debug_frame = self.debug_visualizer.create_debug_frame(
                            original_frame=frame,
                            detections=detections,
                            tracking_results=tracking_results,
                            roi_events=roi_events,
                            sam_results=sam_results,
                            classification_results=classification_results,
                            frame_id=frame_id,
                            roi_monitor=self.roi_monitor,
                            pipeline_stats=self.get_pipeline_status()
                        )
                        
                        # Add legend to first few frames
                        if frame_id < 100:
                            debug_frame = self.debug_visualizer.add_legend(debug_frame)
                        
                        self.debug_video_writer.write_frame(debug_frame)
                    
                    # Update progress
                    progress_bar.set_postfix({
                        'Frame': frame_id,
                        'Count': frame_count,
                        'Objects': len(self.tracker.get_active_tracks()) if self.tracker else 0
                    })
                    
                # Add summary frame to debug video
                if self.debug_visualizer and self.debug_video_writer:
                    summary_frame = self.debug_visualizer.create_summary_frame(self.logger.get_statistics())
                    # Write summary frame multiple times so it's visible
                    for _ in range(int(self.config['video']['fps'] * 3)):  # 3 seconds
                        self.debug_video_writer.write_frame(summary_frame)
                    
                    self.debug_video_writer.close()
                    logger.info("Debug video saved successfully")
                
                # Save all logs
                self.logger.save_logs()
                
                # Print final statistics
                stats = self.logger.get_statistics()
                logger.info(f"Processing completed. Statistics: {stats}")
                
                return True
                
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return False
            
    def _process_frame(self, frame: np.ndarray, frame_id: int, sam_results: Dict, classification_results: Dict):
        """Process a single frame through the pipeline."""
        
        # 1. Object Detection
        detections = self.detector.detect(frame, frame_id)
        self.logger.log_detections(detections, frame_id)
        
        # 2. Object Tracking
        tracking_results = self.tracker.update(detections, frame_id)
        self.logger.log_tracking(tracking_results, frame_id)
        
        # 3. ROI Monitoring
        roi_events = self.roi_monitor.update(tracking_results, frame_id)
        self.logger.log_roi_events(roi_events)
        
        # 4. Process SAM triggers
        for triggered_obj in roi_events.get('triggered', []):
            sam_result, classification_result = self._process_sam_trigger(frame, triggered_obj, frame_id)
            if sam_result:
                sam_results[triggered_obj['track_id']] = sam_result
            if classification_result:
                classification_results[triggered_obj['track_id']] = classification_result
        
        return detections, tracking_results, roi_events
            
    def _process_sam_trigger(self, frame: np.ndarray, triggered_obj: Dict, frame_id: int):
        """Process SAM analysis for triggered object."""
        track_id = triggered_obj['track_id']
        bbox = triggered_obj['bbox']
        initial_class = triggered_obj['class_name']
        
        logger.info(f"Processing SAM trigger for track {track_id} at frame {frame_id}")
        
        # 5. SAM Segmentation
        sam_result = self.sam_segmenter.segment_object(
            frame=frame,
            bbox=bbox,
            track_id=track_id,
            save_path=self.config['video']['output_dir'] + "/masks" if self.config['logging']['save_masks'] else None
        )
        self.logger.log_sam_analysis(sam_result, frame_id)
        
        classification_result = None
        if sam_result['success']:
            # 6. Feature Extraction
            features = self.feature_extractor.extract_features(
                frame=frame,
                mask=sam_result['mask'],
                track_id=track_id
            )
            
            # 7. Classification Refinement
            classification_result = self.classifier.classify_object(
                features=features,
                initial_class=initial_class,
                confidence_threshold=0.3
            )
            self.logger.log_classification(classification_result, frame_id, features)
            
            if classification_result['changed']:
                logger.info(f"Track {track_id} classification changed: "
                          f"{initial_class} → {classification_result['refined_class']} "
                          f"(confidence: {classification_result['confidence']:.3f})")
        
        return sam_result, classification_result
                          
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline processing status."""
        return {
            'current_frame': self.current_frame_id,
            'total_frames': self.total_frames,
            'progress': self.current_frame_id / self.total_frames if self.total_frames > 0 else 0,
            'active_tracks': len(self.tracker.get_active_tracks()) if self.tracker else 0,
            'roi_stats': self.roi_monitor.get_roi_stats() if self.roi_monitor else {},
            'logger_stats': self.logger.get_statistics() if self.logger else {}
        }
        
    def cleanup(self):
        """Clean up pipeline resources."""
        if self.sam_segmenter:
            self.sam_segmenter.cleanup()
            
        if self.debug_video_writer:
            self.debug_video_writer.close()
            
        logger.info("Pipeline cleanup completed")


def main():
    """Main entry point for sam48 pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sam48 Processing Pipeline")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--output", default="output", help="Output directory")
    
    args = parser.parse_args()
    
    # Update output directory in config if provided
    if args.output != "output":
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        config['video']['output_dir'] = args.output
        with open(args.config, 'w') as f:
            yaml.dump(config, f, indent=2)
    
    # Initialize and run pipeline
    pipeline = Sam48Pipeline(args.config)
    
    try:
        success = pipeline.process_video(args.video_path)
        if success:
            logger.info("Pipeline completed successfully")
        else:
            logger.error("Pipeline failed")
            exit(1)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()