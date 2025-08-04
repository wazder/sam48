import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler


class ObjectClassifier:
    """Refine object classification using extracted features and reference data."""
    
    def __init__(self, reference_data_path: Optional[str] = None):
        """
        Initialize object classifier.
        
        Args:
            reference_data_path: Path to reference feature data (JSON file)
        """
        self.reference_data_path = reference_data_path
        self.reference_features = {}
        self.scaler = None
        self.target_classes = ["person", "handbag", "backpack", "suitcase"]
        
        # Load reference data if provided
        if reference_data_path and Path(reference_data_path).exists():
            self._load_reference_data()
        else:
            # Create default reference features based on typical characteristics
            self._create_default_reference_features()
            
        logger.info(f"Object classifier initialized with {len(self.reference_features)} reference classes")
        
    def _load_reference_data(self):
        """Load reference feature data from JSON file."""
        try:
            with open(self.reference_data_path, 'r') as f:
                data = json.load(f)
                
            self.reference_features = data.get('reference_features', {})
            
            # Initialize scaler if normalization data is available
            if 'scaler_params' in data:
                self.scaler = StandardScaler()
                self.scaler.mean_ = np.array(data['scaler_params']['mean'])
                self.scaler.scale_ = np.array(data['scaler_params']['scale'])
                
            logger.info(f"Loaded reference data from {self.reference_data_path}")
            
        except Exception as e:
            logger.error(f"Failed to load reference data: {e}")
            self._create_default_reference_features()
            
    def _create_default_reference_features(self):
        """Create default reference features based on typical object characteristics."""
        # These are simplified reference features based on general knowledge
        # In practice, these would be learned from a dataset
        
        self.reference_features = {
            "person": {
                "color_characteristics": {
                    "dominant_hues": [0, 30, 60, 120],  # Skin tones, clothing colors
                    "saturation_range": [0.2, 0.8],
                    "brightness_range": [0.3, 0.9]
                },
                "shape_characteristics": {
                    "aspect_ratio_range": [0.3, 0.7],  # Typically taller than wide
                    "compactness_range": [0.1, 0.4],   # Irregular shape
                    "solidity_range": [0.7, 0.95],     # Moderate solidity
                    "typical_area_range": [5000, 50000]  # Pixel area
                },
                "texture_characteristics": {
                    "roughness_range": [10, 40],
                    "edge_density_range": [0.1, 0.5]
                }
            },
            "handbag": {
                "color_characteristics": {
                    "dominant_hues": [0, 15, 30, 45, 300],  # Browns, blacks, reds
                    "saturation_range": [0.3, 0.9],
                    "brightness_range": [0.2, 0.8]
                },
                "shape_characteristics": {
                    "aspect_ratio_range": [0.8, 2.0],  # Wider variety
                    "compactness_range": [0.3, 0.7],   # Moderate compactness
                    "solidity_range": [0.8, 0.98],     # High solidity
                    "typical_area_range": [1000, 15000]
                },
                "texture_characteristics": {
                    "roughness_range": [5, 25],
                    "edge_density_range": [0.05, 0.3]
                }
            },
            "backpack": {
                "color_characteristics": {
                    "dominant_hues": [210, 240, 0, 30, 60],  # Blues, blacks, reds, greens
                    "saturation_range": [0.4, 0.9],
                    "brightness_range": [0.2, 0.7]
                },
                "shape_characteristics": {
                    "aspect_ratio_range": [0.6, 1.2],  # More square-ish
                    "compactness_range": [0.4, 0.8],   # Good compactness
                    "solidity_range": [0.75, 0.95],    # High solidity
                    "typical_area_range": [2000, 25000]
                },
                "texture_characteristics": {
                    "roughness_range": [8, 30],
                    "edge_density_range": [0.1, 0.4]
                }
            },
            "suitcase": {
                "color_characteristics": {
                    "dominant_hues": [0, 15, 30, 210, 240],  # Blacks, browns, blues
                    "saturation_range": [0.2, 0.8],
                    "brightness_range": [0.1, 0.6]
                },
                "shape_characteristics": {
                    "aspect_ratio_range": [1.2, 2.5],  # Wider than tall
                    "compactness_range": [0.5, 0.9],   # High compactness
                    "solidity_range": [0.85, 0.98],    # Very high solidity
                    "typical_area_range": [3000, 40000]
                },
                "texture_characteristics": {
                    "roughness_range": [3, 20],
                    "edge_density_range": [0.02, 0.2]
                }
            }
        }
        
        logger.info("Created default reference features")
        
    def classify_object(self, 
                       features: Dict, 
                       initial_class: str,
                       confidence_threshold: float = 0.3) -> Dict:
        """
        Refine object classification using extracted features.
        
        Args:
            features: Extracted features from FeatureExtractor
            initial_class: Initial classification from YOLO
            confidence_threshold: Minimum confidence for classification change
            
        Returns:
            Dictionary with classification results
        """
        classification_result = {
            'track_id': features.get('track_id'),
            'initial_class': initial_class,
            'refined_class': initial_class,
            'confidence': 0.0,
            'confidence_scores': {},
            'reasoning': [],
            'changed': False
        }
        
        try:
            # Calculate similarity scores for each target class
            similarity_scores = {}
            
            for target_class in self.target_classes:
                score = self._calculate_class_similarity(features, target_class)
                similarity_scores[target_class] = score
                
            classification_result['confidence_scores'] = similarity_scores
            
            # Find best matching class
            best_class = max(similarity_scores.keys(), key=lambda k: similarity_scores[k])
            best_score = similarity_scores[best_class]
            
            classification_result['confidence'] = best_score
            
            # Decide whether to change classification
            if best_class != initial_class and best_score > confidence_threshold:
                # Additional validation - check if the difference is significant
                initial_score = similarity_scores.get(initial_class, 0.0)
                score_difference = best_score - initial_score
                
                if score_difference > 0.1:  # Require significant improvement
                    classification_result['refined_class'] = best_class
                    classification_result['changed'] = True
                    classification_result['reasoning'].append(
                        f"Feature analysis suggests {best_class} (score: {best_score:.3f}) "
                        f"over {initial_class} (score: {initial_score:.3f})"
                    )
                    
            # Add detailed reasoning
            self._add_classification_reasoning(classification_result, features, best_class)
            
            logger.debug(f"Classification: {initial_class} -> {classification_result['refined_class']} "
                        f"(confidence: {best_score:.3f})")
                        
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            classification_result['error'] = str(e)
            
        return classification_result
        
    def _calculate_class_similarity(self, features: Dict, target_class: str) -> float:
        """Calculate similarity between extracted features and target class."""
        if target_class not in self.reference_features:
            return 0.0
            
        reference = self.reference_features[target_class]
        total_score = 0.0
        component_count = 0
        
        # Color similarity
        color_score = self._calculate_color_similarity(features, reference)
        if color_score is not None:
            total_score += color_score
            component_count += 1
            
        # Shape similarity  
        shape_score = self._calculate_shape_similarity(features, reference)
        if shape_score is not None:
            total_score += shape_score * 1.2  # Weight shape features more heavily
            component_count += 1.2
            
        # Texture similarity
        texture_score = self._calculate_texture_similarity(features, reference)
        if texture_score is not None:
            total_score += texture_score * 0.8  # Weight texture features less
            component_count += 0.8
            
        if component_count > 0:
            return total_score / component_count
        else:
            return 0.0
            
    def _calculate_color_similarity(self, features: Dict, reference: Dict) -> Optional[float]:
        """Calculate color feature similarity."""
        if 'color_features' not in features:
            return None
            
        color_features = features['color_features']
        color_ref = reference.get('color_characteristics', {})
        
        score = 0.0
        components = 0
        
        # Check dominant colors if available
        if 'dominant_colors' in color_features and 'dominant_hues' in color_ref:
            dom_colors = color_features['dominant_colors']
            ref_hues = color_ref['dominant_hues']
            
            if dom_colors:
                # Convert dominant colors to HSV and extract hues
                hue_matches = 0
                total_colors = len(dom_colors)
                
                for color in dom_colors:
                    # Convert RGB to HSV (approximate)
                    if len(color) >= 3:
                        r, g, b = color[0], color[1], color[2]
                        hsv = self._rgb_to_hsv(r, g, b)
                        hue = hsv[0] * 360  # Convert to degrees
                        
                        # Check if hue matches any reference hue (within tolerance)
                        for ref_hue in ref_hues:
                            if abs(hue - ref_hue) < 30:  # 30 degree tolerance
                                hue_matches += 1
                                break
                                
                if total_colors > 0:
                    score += (hue_matches / total_colors)
                    components += 1
                    
        # Check color statistics
        if 'statistics' in color_features:
            stats = color_features['statistics']
            
            # Check saturation and brightness ranges (assuming HSV color space from FeatureExtractor)
            # This assumes the FeatureExtractor is using HSV color space
            if 'C2' in stats and 'saturation_range' in color_ref:
                sat_mean = stats['C2']['mean'] / 255.0
                sat_range = color_ref['saturation_range']
                if sat_range[0] <= sat_mean <= sat_range[1]:
                    score += 1.0
                else:
                    # Partial score based on distance
                    dist = min(abs(sat_mean - sat_range[0]), abs(sat_mean - sat_range[1]))
                    score += max(0, 1.0 - dist)
                components += 1
                
            if 'C3' in stats and 'brightness_range' in color_ref:
                bright_mean = stats['C3']['mean'] / 255.0
                bright_range = color_ref['brightness_range']
                if bright_range[0] <= bright_mean <= bright_range[1]:
                    score += 1.0
                else:
                    dist = min(abs(bright_mean - bright_range[0]), abs(bright_mean - bright_range[1]))
                    score += max(0, 1.0 - dist)
                components += 1
                    
        return score / components if components > 0 else None
        
    def _calculate_shape_similarity(self, features: Dict, reference: Dict) -> Optional[float]:
        """Calculate shape feature similarity."""
        if 'shape_features' not in features:
            return None
            
        shape_features = features['shape_features']
        shape_ref = reference.get('shape_characteristics', {})
        
        score = 0.0
        components = 0
        
        # Check aspect ratio
        if 'aspect_ratio' in shape_features and 'aspect_ratio_range' in shape_ref:
            aspect_ratio = shape_features['aspect_ratio']
            ar_range = shape_ref['aspect_ratio_range']
            
            if ar_range[0] <= aspect_ratio <= ar_range[1]:
                score += 1.0
            else:
                # Partial score based on distance
                dist = min(abs(aspect_ratio - ar_range[0]), abs(aspect_ratio - ar_range[1]))
                score += max(0, 1.0 - min(dist, 1.0))
            components += 1
            
        # Check compactness
        if 'compactness' in shape_features and 'compactness_range' in shape_ref:
            compactness = shape_features['compactness']
            comp_range = shape_ref['compactness_range']
            
            if comp_range[0] <= compactness <= comp_range[1]:
                score += 1.0
            else:
                dist = min(abs(compactness - comp_range[0]), abs(compactness - comp_range[1]))
                score += max(0, 1.0 - min(dist, 1.0))
            components += 1
            
        # Check solidity
        if 'solidity' in shape_features and 'solidity_range' in shape_ref:
            solidity = shape_features['solidity']
            sol_range = shape_ref['solidity_range']
            
            if sol_range[0] <= solidity <= sol_range[1]:
                score += 1.0
            else:
                dist = min(abs(solidity - sol_range[0]), abs(solidity - sol_range[1]))
                score += max(0, 1.0 - min(dist, 1.0))
            components += 1
            
        # Check area (with logarithmic scaling due to wide range)
        if 'area' in shape_features and 'typical_area_range' in shape_ref:
            area = shape_features['area']
            area_range = shape_ref['typical_area_range']
            
            if area_range[0] <= area <= area_range[1]:
                score += 1.0
            else:
                # Use logarithmic distance for area
                log_area = np.log10(max(area, 1))
                log_range = [np.log10(area_range[0]), np.log10(area_range[1])]
                dist = min(abs(log_area - log_range[0]), abs(log_area - log_range[1]))
                score += max(0, 1.0 - min(dist / 2.0, 1.0))  # Scale down the distance
            components += 1
            
        return score / components if components > 0 else None
        
    def _calculate_texture_similarity(self, features: Dict, reference: Dict) -> Optional[float]:
        """Calculate texture feature similarity."""
        if 'texture_features' not in features:
            return None
            
        texture_features = features['texture_features']
        texture_ref = reference.get('texture_characteristics', {})
        
        score = 0.0
        components = 0
        
        # Check roughness
        if 'roughness' in texture_features and 'roughness_range' in texture_ref:
            roughness = texture_features['roughness']
            rough_range = texture_ref['roughness_range']
            
            if rough_range[0] <= roughness <= rough_range[1]:
                score += 1.0
            else:
                dist = min(abs(roughness - rough_range[0]), abs(roughness - rough_range[1]))
                score += max(0, 1.0 - min(dist / max(rough_range), 1.0))
            components += 1
            
        # Check edge density
        if 'edge_density' in texture_features and 'edge_density_range' in texture_ref:
            edge_density = texture_features['edge_density']
            edge_range = texture_ref['edge_density_range']
            
            if edge_range[0] <= edge_density <= edge_range[1]:
                score += 1.0
            else:
                dist = min(abs(edge_density - edge_range[0]), abs(edge_density - edge_range[1]))
                score += max(0, 1.0 - min(dist / max(edge_range), 1.0))
            components += 1
            
        return score / components if components > 0 else None
        
    def _add_classification_reasoning(self, result: Dict, features: Dict, predicted_class: str):
        """Add detailed reasoning for classification decision."""
        reasoning = result['reasoning']
        
        # Add key feature observations
        if 'shape_features' in features:
            shape = features['shape_features']
            if 'aspect_ratio' in shape:
                ar = shape['aspect_ratio']
                if ar < 0.5:
                    reasoning.append(f"Tall object (aspect ratio: {ar:.2f}) - favors person")
                elif ar > 1.5:
                    reasoning.append(f"Wide object (aspect ratio: {ar:.2f}) - favors suitcase")
                    
            if 'compactness' in shape:
                comp = shape['compactness']
                if comp < 0.3:
                    reasoning.append(f"Irregular shape (compactness: {comp:.2f}) - favors person")
                elif comp > 0.7:
                    reasoning.append(f"Compact shape (compactness: {comp:.2f}) - favors bag/suitcase")
                    
        # Add color observations
        if 'color_features' in features:
            color = features['color_features']
            if 'dominant_colors' in color and color['dominant_colors']:
                dom_color = color['dominant_colors'][0]
                reasoning.append(f"Dominant color: RGB{dom_color}")
                
    def _rgb_to_hsv(self, r: int, g: int, b: int) -> Tuple[float, float, float]:
        """Convert RGB to HSV color space."""
        r, g, b = r/255.0, g/255.0, b/255.0
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val
        
        # Hue
        if diff == 0:
            h = 0
        elif max_val == r:
            h = (60 * ((g - b) / diff) + 360) % 360
        elif max_val == g:
            h = (60 * ((b - r) / diff) + 120) % 360
        else:
            h = (60 * ((r - g) / diff) + 240) % 360
            
        # Saturation
        s = 0 if max_val == 0 else diff / max_val
        
        # Value
        v = max_val
        
        return (h/360.0, s, v)  # Normalize hue to [0, 1]
        
    def save_reference_features(self, output_path: str):
        """Save current reference features to JSON file."""
        data = {
            'reference_features': self.reference_features,
            'target_classes': self.target_classes
        }
        
        if self.scaler is not None:
            data['scaler_params'] = {
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist()
            }
            
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Reference features saved to {output_path}")