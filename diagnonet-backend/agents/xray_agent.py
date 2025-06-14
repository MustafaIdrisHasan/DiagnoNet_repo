import io
import logging
from PIL import Image
import numpy as np
from typing import Dict, Optional, Tuple
import base64

logger = logging.getLogger(__name__)

class XRayAgent:
    """
    X-Ray Image Analysis Agent for medical diagnosis
    
    Processes chest X-ray images and provides analysis for integration
    with the BioGPT supervisor agent.
    """
    
    def __init__(self):
        self.target_size = (512, 512)
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.supported_formats = {'PNG', 'JPEG', 'JPG'}
        
        # Medical findings patterns (simplified for demonstration)
        self.findings_patterns = {
            'normal': {
                'keywords': ['clear', 'normal', 'no abnormalities'],
                'confidence_base': 0.8
            },
            'pneumonia': {
                'keywords': ['consolidation', 'opacity', 'infiltrate'],
                'confidence_base': 0.7
            },
            'pneumothorax': {
                'keywords': ['collapsed lung', 'air space', 'pleural'],
                'confidence_base': 0.75
            },
            'cardiomegaly': {
                'keywords': ['enlarged heart', 'cardiac', 'heart size'],
                'confidence_base': 0.72
            },
            'pleural_effusion': {
                'keywords': ['fluid', 'pleural space', 'effusion'],
                'confidence_base': 0.73
            }
        }
        
        logger.info("X-Ray Analysis Agent initialized")
    
    def validate_image(self, image_data: bytes, filename: str) -> Tuple[bool, str]:
        """
        Validate uploaded image file
        
        Args:
            image_data: Raw image bytes
            filename: Original filename
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file size
            if len(image_data) > self.max_file_size:
                return False, f"File size ({len(image_data)/1024/1024:.1f}MB) exceeds 10MB limit"
            
            # Check if it's a valid image
            try:
                image = Image.open(io.BytesIO(image_data))
                image.verify()  # Verify it's a valid image
            except Exception:
                return False, "Invalid image file format"
            
            # Check format
            image = Image.open(io.BytesIO(image_data))  # Reopen after verify
            if image.format not in self.supported_formats:
                return False, f"Unsupported format. Please use PNG, JPG, or JPEG"
            
            # Basic medical image validation (size and aspect ratio)
            width, height = image.size
            if width < 100 or height < 100:
                return False, "Image too small for medical analysis (minimum 100x100 pixels)"
            
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 3.0:
                return False, "Image aspect ratio suggests it may not be a medical X-ray"
            
            return True, ""
            
        except Exception as e:
            logger.error(f"Image validation error: {str(e)}")
            return False, f"Image validation failed: {str(e)}"
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """
        Preprocess X-ray image for analysis
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Open and convert image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')
            
            # Resize to standard dimensions
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(image, dtype=np.float32)
            img_array = img_array / 255.0  # Normalize to 0-1
            
            return img_array
            
        except Exception as e:
            logger.error(f"Image preprocessing error: {str(e)}")
            raise ValueError(f"Failed to preprocess image: {str(e)}")
    
    def extract_features(self, img_array: np.ndarray) -> Dict:
        """
        Extract basic features from X-ray image
        
        Args:
            img_array: Preprocessed image array
            
        Returns:
            Dictionary of extracted features
        """
        try:
            # Basic statistical features
            mean_intensity = float(np.mean(img_array))
            std_intensity = float(np.std(img_array))
            min_intensity = float(np.min(img_array))
            max_intensity = float(np.max(img_array))
            
            # Contrast and brightness measures
            contrast = std_intensity / mean_intensity if mean_intensity > 0 else 0
            
            # Edge detection (simplified)
            edges = np.abs(np.gradient(img_array))
            edge_density = float(np.mean(edges))
            
            # Symmetry analysis (left vs right lung comparison)
            left_half = img_array[:, :img_array.shape[1]//2]
            right_half = img_array[:, img_array.shape[1]//2:]
            symmetry_score = 1.0 - abs(np.mean(left_half) - np.mean(right_half))
            
            # Lung field estimation (simplified)
            # Assume lung fields are in the middle portion of the image
            lung_region = img_array[int(0.2*img_array.shape[0]):int(0.8*img_array.shape[0]),
                                   int(0.1*img_array.shape[1]):int(0.9*img_array.shape[1])]
            lung_clarity = float(np.std(lung_region))
            
            features = {
                'mean_intensity': mean_intensity,
                'std_intensity': std_intensity,
                'contrast': contrast,
                'edge_density': edge_density,
                'symmetry_score': symmetry_score,
                'lung_clarity': lung_clarity,
                'intensity_range': max_intensity - min_intensity
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            return {}
    
    def analyze_findings(self, features: Dict) -> Dict:
        """
        Analyze X-ray features to generate medical findings
        
        Args:
            features: Extracted image features
            
        Returns:
            Analysis results with findings and confidence
        """
        try:
            findings = []
            overall_confidence = 0.0
            primary_finding = "Normal chest X-ray"
            
            # Analyze based on extracted features
            if features.get('lung_clarity', 0) < 0.1:
                findings.append("Possible consolidation or opacity detected")
                primary_finding = "Abnormal - consolidation suspected"
                overall_confidence = 0.65
                
            elif features.get('symmetry_score', 1.0) < 0.85:
                findings.append("Asymmetric lung fields noted")
                primary_finding = "Abnormal - asymmetric findings"
                overall_confidence = 0.60
                
            elif features.get('contrast', 0) > 0.8:
                findings.append("High contrast areas detected")
                primary_finding = "Abnormal - high contrast regions"
                overall_confidence = 0.58
                
            else:
                findings.append("Lung fields appear clear")
                findings.append("No obvious abnormalities detected")
                primary_finding = "Normal chest X-ray"
                overall_confidence = 0.75
            
            # Add technical quality assessment
            if features.get('intensity_range', 0) < 0.3:
                findings.append("Image quality: Low contrast")
            elif features.get('intensity_range', 0) > 0.8:
                findings.append("Image quality: Good contrast")
            else:
                findings.append("Image quality: Adequate")
            
            analysis_result = {
                'primary_finding': primary_finding,
                'findings': findings,
                'confidence': overall_confidence,
                'technical_quality': 'adequate',
                'features_analyzed': len(features),
                'agent_type': 'xray_analysis'
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return {
                'primary_finding': 'Analysis failed',
                'findings': ['Unable to analyze image'],
                'confidence': 0.0,
                'technical_quality': 'poor',
                'features_analyzed': 0,
                'agent_type': 'xray_analysis'
            }
    
    def analyze_xray(self, image_data: bytes, filename: str) -> Dict:
        """
        Complete X-ray analysis pipeline
        
        Args:
            image_data: Raw image bytes
            filename: Original filename
            
        Returns:
            Complete analysis results
        """
        try:
            logger.info(f"Starting X-ray analysis for file: {filename}")
            
            # Validate image
            is_valid, error_msg = self.validate_image(image_data, filename)
            if not is_valid:
                return {
                    'primary_finding': 'Invalid image',
                    'findings': [error_msg],
                    'confidence': 0.0,
                    'technical_quality': 'invalid',
                    'features_analyzed': 0,
                    'agent_type': 'xray_analysis'
                }
            
            # Preprocess image
            img_array = self.preprocess_image(image_data)
            
            # Extract features
            features = self.extract_features(img_array)
            
            # Analyze findings
            analysis = self.analyze_findings(features)
            
            logger.info(f"X-ray analysis complete: {analysis['primary_finding']}")
            return analysis
            
        except Exception as e:
            logger.error(f"X-ray analysis failed: {str(e)}")
            return {
                'primary_finding': 'Analysis error',
                'findings': [f'Analysis failed: {str(e)}'],
                'confidence': 0.0,
                'technical_quality': 'error',
                'features_analyzed': 0,
                'agent_type': 'xray_analysis'
            }
