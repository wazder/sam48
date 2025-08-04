# Sam48 Processing Pipeline

A comprehensive video processing pipeline that combines object detection, tracking, segmentation, and feature analysis for real-time monitoring applications.

## Features

### Core Pipeline Components
- **YOLO Object Detection**: Detects person, handbag, backpack, suitcase objects
- **Object Tracking**: Assigns unique track IDs using IoU-based tracking
- **Polygon ROI Monitoring**: Monitors objects entering/staying in defined regions
- **Event-Based SAM Analysis**: Triggers segmentation after objects stay in ROI for threshold frames
- **Feature Extraction**: Extracts color, shape, and texture features from segmented objects
- **Classification Refinement**: Refines object classification using extracted features
- **Debug Video Generation**: Creates comprehensive debug video with all processing stages
- **Real-time Info Panel**: Shows frame info, active objects, and processing status
- **Comprehensive Logging**: Saves results in JSON/CSV format with full traceability

### Key Characteristics
- **Event-driven SAM**: Heavy computation only occurs for long-staying objects
- **Polygon-based ROI**: Flexible region definition using line logic
- **Single threshold**: Unified SAM trigger threshold for simplicity
- **Full traceability**: Complete logging of all processing stages
- **Persistent tracking**: Object IDs and classifications tracked over time

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for SAM)

### Setup
```bash
# Clone repository
git clone <repository_url>
cd sam48

# Install dependencies
pip install -r requirements.txt

# Download SAM checkpoint (optional - will use fallback if not available)
# Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints
# Place sam_vit_h_4b8939.pth in project root
```

## Configuration

Edit `config.yaml` to customize the pipeline:

```yaml
video:
  input_resolution: [1024, 1024]
  fps: 30
  output_dir: "output"

detection:
  model: "yolov8n.pt"
  classes: ["person", "handbag", "backpack", "suitcase"]
  confidence_threshold: 0.5

tracking:
  track_thresh: 0.5
  match_thresh: 0.3
  max_disappeared: 30

roi:
  # Define polygon points in clockwise order
  polygon_points: [[300, 300], [700, 300], [700, 700], [300, 700]]

sam:
  model_type: "vit_h"
  checkpoint_path: "sam_vit_h_4b8939.pth"
  threshold: 10  # Frames to stay in ROI before SAM analysis

logging:
  format: "json"  # json or csv
  save_masks: true

debug:
  debug_video: true     # Generate debug video with overlays
  show_bboxes: true     # Show detection bounding boxes
  show_track_ids: true  # Show track IDs
  show_roi: true        # Show ROI visualization
  show_sam_masks: true  # Show SAM segmentation masks
  info_panel: true      # Show real-time info panel
```

## Usage

### Basic Usage
```bash
python main.py input_video.mp4
```

### Advanced Usage
```bash
# Custom config and output directory
python main.py input_video.mp4 --config custom_config.yaml --output results

# Using the pipeline module directly
python -m src.pipeline input_video.mp4
```

## Pipeline Flow

1. **Video Input**: Process 1024×1024 video frame by frame
2. **Object Detection**: YOLO detects target objects with bounding boxes
3. **Object Tracking**: Assign unique track IDs to maintain object identity
4. **ROI Monitoring**: Check if object centers are inside defined polygon
5. **Event Trigger**: After 10 frames in ROI, trigger SAM analysis
6. **SAM Segmentation**: Generate precise object masks using SAM
7. **Feature Extraction**: Extract color, shape, and texture features
8. **Classification Refinement**: Compare features to refine object classification
9. **Logging**: Save all data to JSON/CSV files with complete traceability

## Output Structure

```
output/
├── logs/
│   ├── detections_YYYYMMDD_HHMMSS.json
│   ├── tracking_YYYYMMDD_HHMMSS.json
│   ├── roi_events_YYYYMMDD_HHMMSS.json
│   ├── sam_analysis_YYYYMMDD_HHMMSS.json
│   ├── classification_YYYYMMDD_HHMMSS.json
│   └── session_summary_YYYYMMDD_HHMMSS.json
├── masks/
│   ├── track_1_mask.png
│   ├── track_2_mask.png
│   └── ...
└── videos/
    └── debug_input_video.mp4  # Debug video with all overlays
```

## API Usage

```python
from src.pipeline import Sam48Pipeline

# Initialize pipeline
pipeline = Sam48Pipeline("config.yaml")

# Process video
success = pipeline.process_video("input_video.mp4")

# Get statistics
stats = pipeline.get_pipeline_status()
print(f"Processed {stats['total_frames']} frames")

# Cleanup
pipeline.cleanup()
```

## Components

### Core Modules
- `src/detection/yolo_detector.py` - YOLO object detection
- `src/tracking/simple_tracker.py` - IoU-based object tracking
- `src/utils/roi_monitor.py` - Polygon ROI monitoring
- `src/segmentation/sam_segmenter.py` - SAM segmentation
- `src/features/feature_extractor.py` - Feature extraction
- `src/features/classifier.py` - Classification refinement
- `src/utils/logger.py` - Comprehensive logging
- `src/pipeline.py` - Main pipeline orchestrator

### Key Features
- **Fallback Support**: Works without SAM (uses rectangular masks)
- **GPU Acceleration**: Automatic CUDA detection for SAM
- **Polygon ROI**: Ray-casting algorithm for point-in-polygon detection
- **Feature-based Classification**: Multi-modal feature comparison
- **Comprehensive Logging**: Complete pipeline traceability

## Performance Notes

- SAM analysis is computationally expensive - only triggered for persistent objects
- Use CUDA-enabled GPU for better SAM performance
- Polygon ROI checking is O(n) where n is number of polygon vertices
- Feature extraction includes color histograms, shape descriptors, and texture analysis

## Troubleshooting

### Common Issues
1. **SAM not found**: Install segment-anything or download checkpoint
2. **CUDA out of memory**: Reduce batch size or use CPU
3. **Video not opening**: Check video path and codec support
4. **Poor tracking**: Adjust confidence and IoU thresholds

### Dependencies
- Core: opencv-python, numpy, ultralytics, scikit-learn
- Optional: segment-anything (for precise segmentation)
- Logging: pandas, pyyaml, loguru, tqdm

## License

See LICENSE file for details.
