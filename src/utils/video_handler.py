import cv2
import numpy as np
from typing import Generator, Tuple, Optional
from loguru import logger


class VideoHandler:
    """Handles video input and frame processing for the sam48 pipeline."""
    
    def __init__(self, video_path: str, target_resolution: Tuple[int, int] = (1024, 1024)):
        """
        Initialize video handler.
        
        Args:
            video_path: Path to input video file
            target_resolution: Target resolution (width, height)
        """
        self.video_path = video_path
        self.target_resolution = target_resolution
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.original_resolution = None
        
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        
    def open(self) -> bool:
        """Open video file and initialize properties."""
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open video: {self.video_path}")
            return False
            
        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_resolution = (width, height)
        
        logger.info(f"Video opened: {self.video_path}")
        logger.info(f"Original resolution: {self.original_resolution}")
        logger.info(f"Target resolution: {self.target_resolution}")
        logger.info(f"Total frames: {self.total_frames}, FPS: {self.fps}")
        
        return True
        
    def close(self):
        """Close video file."""
        if self.cap:
            self.cap.release()
            logger.info("Video closed")
            
    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator that yields frame_id and processed frame.
        
        Yields:
            Tuple of (frame_id, frame_array)
        """
        if not self.cap:
            raise RuntimeError("Video not opened. Use open() or context manager.")
            
        frame_id = 0
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                break
                
            # Resize frame to target resolution
            if self.original_resolution != self.target_resolution:
                frame = cv2.resize(frame, self.target_resolution)
                
            yield frame_id, frame
            frame_id += 1
            
    def get_frame_at(self, frame_id: int) -> Optional[np.ndarray]:
        """
        Get specific frame by ID.
        
        Args:
            frame_id: Frame number to retrieve
            
        Returns:
            Frame array or None if frame doesn't exist
        """
        if not self.cap:
            raise RuntimeError("Video not opened")
            
        if frame_id >= self.total_frames:
            return None
            
        # Set frame position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.cap.read()
        
        if not ret:
            return None
            
        # Resize frame to target resolution
        if self.original_resolution != self.target_resolution:
            frame = cv2.resize(frame, self.target_resolution)
            
        return frame
        
    def reset(self):
        """Reset video to beginning."""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
    @property
    def duration_seconds(self) -> float:
        """Get video duration in seconds."""
        return self.total_frames / self.fps if self.fps > 0 else 0
        
    @property
    def is_opened(self) -> bool:
        """Check if video is opened."""
        return self.cap is not None and self.cap.isOpened()


class VideoWriter:
    """Handles debug video output generation."""
    
    def __init__(self, output_path: str, resolution: Tuple[int, int], fps: int = 30):
        """
        Initialize video writer.
        
        Args:
            output_path: Output video file path
            resolution: Video resolution (width, height)
            fps: Frames per second
        """
        self.output_path = output_path
        self.resolution = resolution
        self.fps = fps
        self.writer = None
        
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        
    def open(self):
        """Open video writer."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, self.resolution
        )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.output_path}")
            
        logger.info(f"Video writer opened: {self.output_path}")
        
    def write_frame(self, frame: np.ndarray):
        """Write a frame to the video."""
        if not self.writer:
            raise RuntimeError("Video writer not opened")
            
        self.writer.write(frame)
        
    def close(self):
        """Close video writer."""
        if self.writer:
            self.writer.release()
            logger.info(f"Video saved: {self.output_path}")