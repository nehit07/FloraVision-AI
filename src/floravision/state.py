"""
FloraVision AI - State Models
=============================

This is the FOUNDATION of the entire system.

PURPOSE:
    Defines the data structure (PlantState) that flows through 
    all 8 reasoning nodes in the LangGraph pipeline.

CONNECTIONS:
    ┌─────────────────────────────────────────────────────────────┐
    │                      PlantState                             │
    │   (This file defines the central data contract)            │
    │                          │                                  │
    │   ┌──────────────────────┼──────────────────────┐           │
    │   │                      │                      │           │
    │   ▼                      ▼                      ▼           │
    │ graph.py             nodes/*              detection/*       │
    │ (creates state)   (read & update)      (populates state)   │
    └─────────────────────────────────────────────────────────────┘

USAGE:
    from floravision.state import PlantState, YOLODetection
    
    state = PlantState(
        image=image_bytes,
        season="winter"
    )
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# Knowledge base version for reproducibility
KNOWLEDGE_VERSION = "1.0.0"


class YOLODetection(BaseModel):
    """
    Represents a single YOLO detection result.
    
    Attributes:
        label: The detected symptom (e.g., "leaf_yellowing", "brown_spots")
        confidence: Detection confidence score (0.0 to 1.0)
    
    Used by:
        - detection/yolo_detector.py (creates these)
        - nodes/symptoms.py (reads these)
        - nodes/severity.py (uses confidence for calculation)
    """
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: Optional[List[float]] = None  # [x1, y1, x2, y2] normalized or pixel coordinates


class PlantState(BaseModel):
    """
    Central state object that flows through the LangGraph pipeline.
    
    This is the SINGLE SOURCE OF TRUTH for the entire diagnosis.
    Every node reads from and writes to this state.
    
    Flow:
        1. app.py creates initial state with image + season
        2. detection/yolo_detector.py fills yolo_detections
        3. detection/plant_id.py fills plant_name + plant_id_confidence
        4. Each node (1-8) adds its specific fields
        5. Final state contains complete diagnosis
    """
    
    image: Optional[bytes] = None
    
    season: str = "unknown"
    climate_zone: str = "Temperate"
    plant_name: str = "Unknown"
    plant_id_confidence: float = 0.0
    
    yolo_detections: List[YOLODetection] = Field(default_factory=list)
    symptoms_grouped: Dict[str, List[str]] = Field(default_factory=dict)

    severity: Optional[str] = None
    diagnosis_confidence: Optional[str] = None
    is_healthy: bool = False

    causes: List[str] = Field(default_factory=list)
    seasonal_insight: Optional[str] = None
    care_immediate: List[str] = Field(default_factory=list)
    care_ongoing: List[str] = Field(default_factory=list)
    care_calendar: List[Dict[str, str]] = Field(default_factory=list)
    
    dont_do: List[str] = Field(default_factory=list)
    
    pro_tip: Optional[str] = None
    
    rescan_suggested: bool = False
    
    final_response: Optional[str] = None
    reasoning_trace: List[str] = Field(default_factory=list)


def calculate_confidence(state: PlantState) -> str:
    """
    Calculate overall diagnosis confidence based on multiple factors.
    
    Used by: nodes/severity.py
    
    Factors:
        1. Plant ID confidence (above 70%?)
        2. Number of YOLO detections (at least 2?)
        3. YOLO detection confidence (all above 50%?)
    
    Returns:
        "High", "Medium", or "Low"
    """
    factors = [
        state.plant_id_confidence > 0.7,
        len(state.yolo_detections) >= 2,
        all(d.confidence > 0.5 for d in state.yolo_detections) if state.yolo_detections else False
    ]
    score = sum(factors)
    
    if score >= 3:
        return "High"
    elif score >= 2:
        return "Medium"
    return "Low"
