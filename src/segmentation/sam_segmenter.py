import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union
import torch
from pathlib import Path
from loguru import logger

try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    logger.warning("Segment Anything not available. Install with: pip install segment-anything")
    SAM_AVAILABLE = False


class SAMSegmenter:
    """Segment Anything Model (SAM) segmentation for triggered objects."""
    
    def __init__(self, 
                 model_type: str = "vit_h",
                 checkpoint_path: str = "sam_vit_h_4b8939.pth",
                 device: str = "auto"):
        """
        Initialize SAM segmenter.
        
        Args:
            model_type: SAM model type ("vit_h", "vit_l", "vit_b")
            checkpoint_path: Path to SAM checkpoint file
            device: Device to run on ("cuda", "cpu", or "auto")
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.predictor = None
        self.is_initialized = False
        
        logger.info(f"SAM Segmenter initialized: {model_type} on {self.device}")
        
    def initialize(self) -> bool:
        """Initialize SAM model (lazy loading)."""
        if not SAM_AVAILABLE:
            logger.error("Segment Anything not available")
            return False
            
        if self.is_initialized:
            return True
            
        try:
            # Check if checkpoint exists
            if not Path(self.checkpoint_path).exists():
                logger.error(f"SAM checkpoint not found: {self.checkpoint_path}")
                logger.info("Download SAM checkpoints from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
                return False
                
            # Load SAM model
            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            sam.to(device=self.device)
            
            # Create predictor
            self.predictor = SamPredictor(sam)
            self.is_initialized = True
            
            logger.info(f"SAM model loaded successfully: {self.checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SAM: {e}")
            return False
            
    def segment_object(self, 
                      frame: np.ndarray, 
                      bbox: List[float],
                      track_id: int,
                      save_path: Optional[str] = None) -> Dict:
        """
        Segment object using SAM based on bounding box.
        
        Args:
            frame: Input frame (BGR format)
            bbox: Bounding box [x1, y1, x2, y2]
            track_id: Track ID for naming
            save_path: Optional path to save mask image
            
        Returns:
            Dictionary with segmentation results
        """
        if not self.initialize():
            return self._create_fallback_mask(frame, bbox, track_id, save_path)
            
        try:
            # Convert BGR to RGB for SAM
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Set image for SAM predictor
            self.predictor.set_image(frame_rgb)
            
            # Convert bbox to center point and dimensions for better prompting
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Use center point as input prompt
            input_point = np.array([[center_x, center_y]])
            input_label = np.array([1])  # Foreground point
            
            # Also use bounding box as additional prompt
            input_box = np.array([x1, y1, x2, y2])
            
            # Generate masks
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                multimask_output=True
            )
            
            # Select best mask (highest score)
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            best_score = scores[best_mask_idx]
            
            # Create masked image on black background
            masked_image = self._create_masked_image(frame, best_mask)
            
            # Save mask if path provided
            mask_path = None
            if save_path:
                mask_path = self._save_mask(masked_image, save_path, track_id)
            
            # Calculate mask statistics
            mask_stats = self._calculate_mask_stats(best_mask)
            
            result = {
                'success': True,
                'track_id': track_id,
                'mask': best_mask,
                'score': float(best_score),
                'masked_image': masked_image,
                'mask_path': mask_path,
                'stats': mask_stats,
                'bbox': bbox,
                'center_point': [center_x, center_y]
            }
            
            logger.debug(f"SAM segmentation successful for track {track_id}, score: {best_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"SAM segmentation failed for track {track_id}: {e}")
            return self._create_fallback_mask(frame, bbox, track_id, save_path)
            
    def _create_fallback_mask(self, 
                             frame: np.ndarray, 
                             bbox: List[float], 
                             track_id: int,
                             save_path: Optional[str] = None) -> Dict:
        """Create fallback rectangular mask when SAM is not available."""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Create rectangular mask
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=bool)
        mask[y1:y2, x1:x2] = True
        
        # Create masked image
        masked_image = self._create_masked_image(frame, mask)
        
        # Save mask if path provided
        mask_path = None
        if save_path:
            mask_path = self._save_mask(masked_image, save_path, track_id)
            
        # Calculate mask statistics
        mask_stats = self._calculate_mask_stats(mask)
        
        result = {
            'success': True,
            'track_id': track_id,
            'mask': mask,
            'score': 1.0,  # Perfect score for rectangular mask
            'masked_image': masked_image,
            'mask_path': mask_path,
            'stats': mask_stats,
            'bbox': bbox,
            'fallback': True  # Indicate this is a fallback mask
        }
        
        logger.debug(f"Created fallback rectangular mask for track {track_id}")
        return result
        
    def _create_masked_image(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create masked image on black background."""
        # Create black background
        masked_image = np.zeros_like(frame)
        
        # Apply mask to each channel
        for c in range(frame.shape[2]):
            masked_image[:, :, c] = frame[:, :, c] * mask
            
        return masked_image
        
    def _save_mask(self, masked_image: np.ndarray, save_dir: str, track_id: int) -> str:
        """Save masked image to file."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"track_{track_id}_mask.png"
        mask_path = save_dir / filename
        
        cv2.imwrite(str(mask_path), masked_image)
        return str(mask_path)
        
    def _calculate_mask_stats(self, mask: np.ndarray) -> Dict:
        """Calculate statistics for the mask."""
        mask_area = np.sum(mask)
        mask_height, mask_width = mask.shape
        total_area = mask_height * mask_width
        
        # Find mask bounds
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if np.any(rows) and np.any(cols):
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            bounding_area = (y_max - y_min + 1) * (x_max - x_min + 1)
            solidity = mask_area / bounding_area if bounding_area > 0 else 0
        else:
            solidity = 0
            
        # Calculate perimeter (approximate)
        perimeter = self._calculate_perimeter(mask)
        
        stats = {
            'area': int(mask_area),
            'perimeter': float(perimeter),
            'solidity': float(solidity),
            'extent': float(mask_area / total_area) if total_area > 0 else 0,
            'compactness': float(4 * np.pi * mask_area / (perimeter ** 2)) if perimeter > 0 else 0
        }
        
        return stats
        
    def _calculate_perimeter(self, mask: np.ndarray) -> float:
        """Calculate approximate perimeter of mask."""
        # Convert to uint8 for OpenCV
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Sum perimeters of all contours
            total_perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
            return total_perimeter
        else:
            return 0.0
            
    def visualize_mask(self, 
                      frame: np.ndarray, 
                      mask: np.ndarray,
                      alpha: float = 0.5,
                      color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Visualize mask overlay on frame.
        
        Args:
            frame: Original frame
            mask: Boolean mask
            alpha: Overlay transparency
            color: Mask color (BGR)
            
        Returns:
            Frame with mask overlay
        """
        overlay = frame.copy()
        
        # Apply color to mask
        overlay[mask] = color
        
        # Blend with original frame
        result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        return result
        
    def cleanup(self):
        """Clean up resources."""
        if self.predictor is not None:
            # Clear GPU memory
            if hasattr(self.predictor, 'model'):
                del self.predictor.model
            del self.predictor
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.predictor = None
        self.is_initialized = False
        logger.info("SAM resources cleaned up")