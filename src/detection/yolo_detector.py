import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from loguru import logger


class YOLODetector:
    """YOLO-based object detector for specified classes."""
    
    # COCO class mappings for target objects
    COCO_CLASS_MAPPING = {
        0: "person",
        24: "handbag", 
        26: "backpack",
        28: "suitcase"
    }
    
    def __init__(self, 
                 model_path: str = "yolov8n.pt",
                 target_classes: List[str] = None,
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.5):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model weights
            target_classes: List of target class names
            confidence_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = model_path
        self.target_classes = target_classes or ["person", "handbag", "backpack", "suitcase"]
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Load YOLO model
        self.model = YOLO(model_path)
        logger.info(f"YOLO model loaded: {model_path}")
        
        # Get target class IDs from COCO mapping
        self.target_class_ids = []
        for class_id, class_name in self.COCO_CLASS_MAPPING.items():
            if class_name in self.target_classes:
                self.target_class_ids.append(class_id)
                
        logger.info(f"Target classes: {self.target_classes}")
        logger.info(f"Target class IDs: {self.target_class_ids}")
        
    def detect(self, frame: np.ndarray, frame_id: int) -> List[Dict]:
        """
        Detect objects in frame.
        
        Args:
            frame: Input frame (BGR format)
            frame_id: Frame identifier
            
        Returns:
            List of detection dictionaries with keys:
            - frame_id: Frame identifier
            - bbox: Bounding box [x1, y1, x2, y2]
            - class_name: Detected class name
            - class_id: COCO class ID
            - confidence: Detection confidence score
        """
        # Run YOLO inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.target_class_ids,
            verbose=False
        )
        
        detections = []
        
        # Process results
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    # Map COCO class ID to class name
                    class_name = self.COCO_CLASS_MAPPING.get(class_id, "unknown")
                    
                    if class_name in self.target_classes:
                        detection = {
                            "frame_id": frame_id,
                            "bbox": box.tolist(),  # [x1, y1, x2, y2]
                            "class_name": class_name,
                            "class_id": int(class_id),
                            "confidence": float(conf)
                        }
                        detections.append(detection)
        
        logger.debug(f"Frame {frame_id}: {len(detections)} detections")
        return detections
    
    def get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """
        Calculate center point of bounding box.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Center point (cx, cy)
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return cx, cy
    
    def get_bbox_area(self, bbox: List[float]) -> float:
        """
        Calculate area of bounding box.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Bounding box area
        """
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
        
    def filter_detections_by_area(self, 
                                  detections: List[Dict], 
                                  min_area: float = 100.0) -> List[Dict]:
        """
        Filter detections by minimum bounding box area.
        
        Args:
            detections: List of detection dictionaries
            min_area: Minimum bounding box area
            
        Returns:
            Filtered detections
        """
        filtered = []
        for detection in detections:
            area = self.get_bbox_area(detection["bbox"])
            if area >= min_area:
                filtered.append(detection)
                
        return filtered
        
    def visualize_detections(self, 
                           frame: np.ndarray, 
                           detections: List[Dict],
                           show_confidence: bool = True,
                           color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw detection bounding boxes on frame.
        
        Args:
            frame: Input frame
            detections: List of detections
            show_confidence: Whether to show confidence scores
            color: Bounding box color (BGR)
            
        Returns:
            Frame with drawn bounding boxes
        """
        import cv2
        
        frame_copy = frame.copy()
        
        for detection in detections:
            bbox = detection["bbox"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}"
            if show_confidence:
                label += f" {confidence:.2f}"
                
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw text background
            cv2.rectangle(
                frame_copy,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                frame_copy,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
        return frame_copy