"""
FloraVision AI - YOLO Symptom Detector
=======================================

PURPOSE:
    Uses YOLOv8 to detect plant health symptoms in images.
    Returns a list of YOLODetection objects with labels and confidence.

CONNECTIONS:
    ┌────────────────────────────────────────────────────────────────┐
    │                    YOLO Detector                               │
    │                         │                                      │
    │   Reads:                │    Writes to:                        │
    │   - Image from state    │    - state.yolo_detections           │
    │                         │                                      │
    │   Used by:              │    Uses:                             │
    │   - graph.py (before    │    - ultralytics (YOLO)              │
    │     pipeline starts)    │    - knowledge/symptoms.json         │
    └────────────────────────────────────────────────────────────────┘

NOTES:
    - For demo/development, we use a mock detector that simulates YOLO
    - For production, integrate a trained YOLO model for plant diseases
    - Model can be from: PlantDoc dataset, PlantVillage, or custom trained
"""

import json
import random
import hashlib
from pathlib import Path
from typing import List, Optional
from PIL import Image
import io

from ..state import YOLODetection


# Path to symptoms knowledge base
SYMPTOMS_PATH = Path(__file__).parent.parent / "knowledge" / "symptoms.json"


class YOLODetector:
    """
    Plant symptom detector using YOLO v8.
    
    For demo purposes, includes a mock mode that simulates detections.
    In production, load a real YOLO model trained on plant diseases.
    
    Usage:
        detector = YOLODetector(mock=True)  # Demo mode
        detections = detector.detect(image_bytes)
    """
    
    def __init__(self, model_path: Optional[str] = None, mock: bool = True):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to trained YOLO model (.pt file)
            mock: If True, use mock detections for demo
        """
        self.mock = mock
        self.model = None
        
        # Load symptom labels from knowledge base
        with open(SYMPTOMS_PATH) as f:
            self.symptom_data = json.load(f)
        self.valid_labels = list(self.symptom_data.keys())
        
        if not mock and model_path:
            # Load real YOLO model
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                self.mock = False
            except Exception as e:
                print(f"Warning: Could not load YOLO model: {e}")
                print("Falling back to mock mode")
                self.mock = True
    
    def detect(self, image_bytes: bytes) -> List[YOLODetection]:
        """
        Detect plant symptoms in the given image.
        
        Args:
            image_bytes: Raw image bytes (from camera/upload)
            
        Returns:
            List of YOLODetection with label and confidence
        """
        if self.mock:
            return self._mock_detect(image_bytes)
        else:
            return self._real_detect(image_bytes)
    
    def _mock_detect(self, image_bytes: bytes) -> List[YOLODetection]:
        """
        Mock detection for demo/development.
        
        Generates deterministic detections based on image hash.
        This ensures the same image always yields the same results.
        """
        detections = []
        
        # Create a deterministic seed from image bytes
        seed = int(hashlib.sha256(image_bytes).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        
        # Analyze image to make mock more realistic
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            # Random but deterministic detections
            num_detections = rng.choices([0, 1, 2, 3], weights=[0.2, 0.4, 0.3, 0.1])[0]
            
            if num_detections == 0:
                # Healthy plant - no symptoms
                return []
            
            # Select deterministic symptoms
            selected_symptoms = rng.sample(
                self.valid_labels, 
                min(num_detections, len(self.valid_labels))
            )
            
            for label in selected_symptoms:
                confidence = rng.uniform(0.55, 0.95)
                # Generate a mock box: [x1, y1, x2, y2]
                x1 = rng.uniform(0.1, 0.6)
                y1 = rng.uniform(0.1, 0.6)
                x2 = x1 + rng.uniform(0.1, 0.3)
                y2 = y1 + rng.uniform(0.1, 0.3)
                
                detections.append(YOLODetection(
                    label=label,
                    confidence=round(confidence, 2),
                    box=[round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                ))
        
        except Exception:
            # If image processing fails, return a default detection
            detections.append(YOLODetection(
                label="leaf_yellowing",
                confidence=0.7
            ))
        
        return detections
    
    def _real_detect(self, image_bytes: bytes) -> List[YOLODetection]:
        """
        Real YOLO detection using trained model.
        
        This requires a YOLO model trained on plant disease dataset.
        """
        detections = []
        
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Run YOLO inference
            if self.model is None:
                print("Warning: YOLO model is None, falling back to mock detections.")
                return self._mock_detect(image_bytes)
                
            results = self.model(image)
            
            for result in results:
                for box in result.boxes:
                    label_idx = int(box.cls)
                    label = result.names[label_idx]
                    confidence = float(box.conf)
                    
                    # Only include if label is in our symptom database
                    if label in self.valid_labels:
                        # Extract coordinates (assuming xyxy format)
                        x1, y1, x2, y2 = box.xyxyn[0].tolist()  # Normalized coordinates
                        
                        detections.append(YOLODetection(
                            label=label,
                            confidence=round(confidence, 2),
                            box=[round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]
                        ))
        
        except Exception as e:
            print(f"YOLO detection error: {e}")
        
        return detections


# Convenience function for quick detection
def detect_symptoms(image_bytes: bytes, mock: bool = True) -> List[YOLODetection]:
    """
    Quick detection function without managing detector instance.
    
    Args:
        image_bytes: Raw image bytes
        mock: Use mock mode for demo
        
    Returns:
        List of detected symptoms
    """
    detector = YOLODetector(mock=mock)
    return detector.detect(image_bytes)
