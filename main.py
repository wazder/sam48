#!/usr/bin/env python3
"""
Sam48 Processing Pipeline - Main Entry Point

A comprehensive video processing pipeline that combines:
- YOLO object detection for person, handbag, backpack, suitcase
- Object tracking with unique track IDs
- Region of Interest (ROI) monitoring with polygon-based detection
- Event-based SAM (Segment Anything Model) analysis
- Color and shape feature extraction
- Object classification refinement
- Comprehensive logging and data storage

Usage:
    python main.py <video_path> [--config config.yaml] [--output output_dir]

Example:
    python main.py input_video.mp4 --config config.yaml --output results
"""

from src.pipeline import Sam48Pipeline
from loguru import logger
import sys


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Video path is required")
        print("Usage: python main.py <video_path> [--config config.yaml] [--output output_dir]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Parse optional arguments
    config_path = "config.yaml"
    output_dir = "output"
    
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
        elif arg == "--output" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
    
    logger.info("=== Sam48 Processing Pipeline ===")
    logger.info(f"Input video: {video_path}")
    logger.info(f"Config file: {config_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Update output directory in config if specified
    if output_dir != "output":
        import yaml
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            config['video']['output_dir'] = output_dir
            with open(config_path, 'w') as f:
                yaml.dump(config, f, indent=2)
            logger.info(f"Updated output directory to: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
    
    # Frame limit (set your desired number here)
    MAX_FRAMES = 100  # Process only first 100 frames
    
    # Initialize and run pipeline
    pipeline = Sam48Pipeline(config_path)
    
    try:
        logger.info("Starting video processing...")
        success = pipeline.process_video(video_path, max_frames=MAX_FRAMES)
        
        if success:
            logger.success("✅ Pipeline completed successfully!")
            
            # Print final statistics
            stats = pipeline.get_pipeline_status()
            logger.info("=== Final Statistics ===")
            logger.info(f"Total frames processed: {stats['total_frames']}")
            logger.info(f"Active tracks: {stats['active_tracks']}")
            logger.info(f"ROI statistics: {stats['roi_stats']}")
            logger.info(f"Logger statistics: {stats['logger_stats']}")
            
        else:
            logger.error("❌ Pipeline failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        sys.exit(1)
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()