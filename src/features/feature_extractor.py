import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from loguru import logger


class FeatureExtractor:
    """Extract color and shape features from segmented objects."""
    
    def __init__(self, color_bins: int = 32, color_space: str = "HSV"):
        """
        Initialize feature extractor.
        
        Args:
            color_bins: Number of bins for color histogram
            color_space: Color space for analysis ("HSV", "BGR", "LAB")
        """
        self.color_bins = color_bins
        self.color_space = color_space
        
        logger.info(f"Feature extractor initialized: {color_bins} bins, {color_space} color space")
        
    def extract_features(self, 
                        frame: np.ndarray, 
                        mask: np.ndarray,
                        track_id: int) -> Dict:
        """
        Extract comprehensive features from masked object.
        
        Args:
            frame: Original frame (BGR format)
            mask: Boolean mask of the object
            track_id: Track identifier
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {
            'track_id': track_id,
            'color_features': {},
            'shape_features': {},
            'texture_features': {},
            'combined_features': {}
        }
        
        try:
            # Extract color features
            features['color_features'] = self._extract_color_features(frame, mask)
            
            # Extract shape features  
            features['shape_features'] = self._extract_shape_features(mask)
            
            # Extract texture features
            features['texture_features'] = self._extract_texture_features(frame, mask)
            
            # Create combined feature vector
            features['combined_features'] = self._create_combined_features(features)
            
            logger.debug(f"Features extracted for track {track_id}")
            
        except Exception as e:
            logger.error(f"Feature extraction failed for track {track_id}: {e}")
            features['error'] = str(e)
            
        return features
        
    def _extract_color_features(self, frame: np.ndarray, mask: np.ndarray) -> Dict:
        """Extract color-based features."""
        color_features = {}
        
        # Convert to specified color space
        if self.color_space == "HSV":
            frame_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        elif self.color_space == "LAB":
            frame_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        else:
            frame_converted = frame  # Keep BGR
            
        # Extract masked pixels
        masked_pixels = frame_converted[mask]
        
        if len(masked_pixels) == 0:
            return {'error': 'No pixels in mask'}
            
        # Color histogram for each channel
        histograms = []
        for channel in range(frame_converted.shape[2]):
            channel_pixels = masked_pixels[:, channel]
            hist, _ = np.histogram(channel_pixels, bins=self.color_bins, range=(0, 256))
            hist = hist.astype(np.float32)
            hist = hist / (np.sum(hist) + 1e-10)  # Normalize
            histograms.append(hist.tolist())
            
        color_features['histograms'] = histograms
        
        # Dominant colors using K-means
        dominant_colors = self._extract_dominant_colors(masked_pixels, n_colors=3)
        color_features['dominant_colors'] = dominant_colors
        
        # Color statistics
        color_stats = {}
        for i, channel_name in enumerate(['C1', 'C2', 'C3']):  # Generic channel names
            channel_pixels = masked_pixels[:, i]
            color_stats[channel_name] = {
                'mean': float(np.mean(channel_pixels)),
                'std': float(np.std(channel_pixels)),
                'median': float(np.median(channel_pixels)),
                'min': int(np.min(channel_pixels)),
                'max': int(np.max(channel_pixels))
            }
        color_features['statistics'] = color_stats
        
        # Color moments (first 3 moments)
        moments = []
        for channel in range(frame_converted.shape[2]):
            channel_pixels = masked_pixels[:, channel].astype(np.float32)
            
            # Normalize to [0,1]
            if np.max(channel_pixels) > 0:
                channel_pixels = channel_pixels / 255.0
                
            # First moment (mean)
            m1 = np.mean(channel_pixels)
            
            # Second moment (variance)
            m2 = np.var(channel_pixels)
            
            # Third moment (skewness)
            m3 = np.mean((channel_pixels - m1) ** 3)
            
            moments.extend([float(m1), float(m2), float(m3)])
            
        color_features['moments'] = moments
        
        return color_features
        
    def _extract_shape_features(self, mask: np.ndarray) -> Dict:
        """Extract shape-based features."""
        shape_features = {}
        
        # Convert mask to uint8 for OpenCV
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'error': 'No contours found'}
            
        # Use largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Basic shape measurements
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        shape_features['area'] = float(area)
        shape_features['perimeter'] = float(perimeter)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        shape_features['bounding_rect'] = [int(x), int(y), int(w), int(h)]
        shape_features['aspect_ratio'] = float(w / h) if h > 0 else 0
        
        # Minimum enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
        shape_features['min_circle'] = {
            'center': [float(cx), float(cy)],
            'radius': float(radius)
        }
        
        # Fitted ellipse (if enough points)
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            (cx, cy), (ma, MA), angle = ellipse
            shape_features['fitted_ellipse'] = {
                'center': [float(cx), float(cy)],
                'axes': [float(ma), float(MA)],
                'angle': float(angle)
            }
        
        # Shape descriptors
        if area > 0 and perimeter > 0:
            # Compactness (circularity)
            compactness = 4 * np.pi * area / (perimeter ** 2)
            shape_features['compactness'] = float(compactness)
            
            # Extent (ratio of contour area to bounding rectangle area)
            extent = area / (w * h) if (w * h) > 0 else 0
            shape_features['extent'] = float(extent)
            
            # Solidity (ratio of contour area to convex hull area)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            shape_features['solidity'] = float(solidity)
            
        # Hu moments (7 invariant moments)
        moments = cv2.moments(largest_contour)
        hu_moments = cv2.HuMoments(moments)
        # Use log transform to make values more manageable
        hu_moments_log = [-np.sign(hu) * np.log10(np.abs(hu) + 1e-10) for hu in hu_moments.flatten()]
        shape_features['hu_moments'] = [float(h) for h in hu_moments_log]
        
        return shape_features
        
    def _extract_texture_features(self, frame: np.ndarray, mask: np.ndarray) -> Dict:
        """Extract texture-based features using Local Binary Patterns."""
        texture_features = {}
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Extract masked region
            masked_gray = gray * mask.astype(np.uint8)
            
            # Simple texture measures
            # Standard deviation as roughness measure
            masked_pixels = gray[mask]
            if len(masked_pixels) > 0:
                texture_features['roughness'] = float(np.std(masked_pixels))
                texture_features['contrast'] = float(np.max(masked_pixels) - np.min(masked_pixels))
                
                # Edge density (using Sobel)
                sobel_x = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(masked_gray, cv2.CV_64F, 0, 1, ksize=3)
                edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                edge_density = np.sum(edge_magnitude[mask]) / np.sum(mask)
                texture_features['edge_density'] = float(edge_density)
            else:
                texture_features['error'] = 'No masked pixels for texture analysis'
                
        except Exception as e:
            texture_features['error'] = f'Texture extraction failed: {e}'
            
        return texture_features
        
    def _extract_dominant_colors(self, pixels: np.ndarray, n_colors: int = 3) -> List[List[int]]:
        """Extract dominant colors using K-means clustering."""
        if len(pixels) == 0:
            return []
            
        try:
            # Reshape pixels for K-means
            pixels_reshaped = pixels.reshape(-1, 3)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=min(n_colors, len(pixels_reshaped)), 
                          random_state=42, n_init=10)
            kmeans.fit(pixels_reshaped)
            
            # Get cluster centers (dominant colors)
            dominant_colors = kmeans.cluster_centers_.astype(int)
            
            return dominant_colors.tolist()
            
        except Exception as e:
            logger.warning(f"Dominant color extraction failed: {e}")
            return []
            
    def _create_combined_features(self, features: Dict) -> Dict:
        """Create combined feature vector from all feature types."""
        combined = {}
        
        try:
            # Flatten color features
            color_vector = []
            if 'histograms' in features['color_features']:
                for hist in features['color_features']['histograms']:
                    color_vector.extend(hist)
            if 'moments' in features['color_features']:
                color_vector.extend(features['color_features']['moments'])
                
            combined['color_vector'] = color_vector
            combined['color_dim'] = len(color_vector)
            
            # Flatten shape features
            shape_vector = []
            shape_features = features['shape_features']
            
            # Add scalar shape features
            scalar_features = ['area', 'perimeter', 'aspect_ratio', 'compactness', 
                             'extent', 'solidity']
            for feature in scalar_features:
                if feature in shape_features:
                    shape_vector.append(shape_features[feature])
                    
            # Add Hu moments
            if 'hu_moments' in shape_features:
                shape_vector.extend(shape_features['hu_moments'])
                
            combined['shape_vector'] = shape_vector
            combined['shape_dim'] = len(shape_vector)
            
            # Flatten texture features
            texture_vector = []
            texture_features = features['texture_features']
            
            texture_scalar_features = ['roughness', 'contrast', 'edge_density']
            for feature in texture_scalar_features:
                if feature in texture_features:
                    texture_vector.append(texture_features[feature])
                    
            combined['texture_vector'] = texture_vector
            combined['texture_dim'] = len(texture_vector)
            
            # Create full combined vector
            full_vector = color_vector + shape_vector + texture_vector
            combined['full_vector'] = full_vector
            combined['full_dim'] = len(full_vector)
            
        except Exception as e:
            logger.error(f"Combined feature creation failed: {e}")
            combined['error'] = str(e)
            
        return combined
        
    def compare_features(self, features1: Dict, features2: Dict) -> Dict:
        """Compare two feature sets and return similarity metrics."""
        comparison = {}
        
        try:
            # Compare color histograms using chi-square distance
            if ('histograms' in features1['color_features'] and 
                'histograms' in features2['color_features']):
                
                hist1 = np.array(features1['color_features']['histograms']).flatten()
                hist2 = np.array(features2['color_features']['histograms']).flatten()
                
                # Chi-square distance
                chi_square = np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-10))
                comparison['color_chi_square'] = float(chi_square)
                
                # Histogram intersection
                intersection = np.sum(np.minimum(hist1, hist2))
                comparison['color_intersection'] = float(intersection)
                
            # Compare shape features using Euclidean distance
            if ('full_vector' in features1['combined_features'] and 
                'full_vector' in features2['combined_features']):
                
                vec1 = np.array(features1['combined_features']['full_vector'])
                vec2 = np.array(features2['combined_features']['full_vector'])
                
                if len(vec1) == len(vec2):
                    # Normalize vectors
                    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
                    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
                    
                    # Euclidean distance
                    euclidean_dist = np.linalg.norm(vec1_norm - vec2_norm)
                    comparison['euclidean_distance'] = float(euclidean_dist)
                    
                    # Cosine similarity
                    cosine_sim = np.dot(vec1_norm, vec2_norm)
                    comparison['cosine_similarity'] = float(cosine_sim)
                    
        except Exception as e:
            logger.error(f"Feature comparison failed: {e}")
            comparison['error'] = str(e)
            
        return comparison