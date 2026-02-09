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
    
    # ═══════════════════════════════════════════════════════════════
    # INPUT FIELDS (set by app.py before pipeline starts)
    # ═══════════════════════════════════════════════════════════════
    image: Optional[bytes] = None
    """Raw image bytes from camera/upload"""
    
    season: str = "unknown"
    """Current season: spring, summer, autumn, winter"""
    
    # ═══════════════════════════════════════════════════════════════
    # DETECTION FIELDS (set by detection/ modules)
    # ═══════════════════════════════════════════════════════════════
    plant_name: str = "Unknown"
    """Identified plant species (set by plant_id.py)"""
    
    plant_id_confidence: float = 0.0
    """Confidence of plant identification (0.0 to 1.0)"""
    
    yolo_detections: List[YOLODetection] = []
    """List of detected symptoms with confidence (set by yolo_detector.py)"""
    
    # ═══════════════════════════════════════════════════════════════
    # NODE 2: SYMPTOM MAPPING (set by nodes/symptoms.py)
    # ═══════════════════════════════════════════════════════════════
    symptoms_grouped: Dict[str, List[str]] = {}
    """
    Symptoms organized by category.
    Example: {"nutrient": ["yellowing"], "fungal": ["brown_spots"]}
    """
    
    # ═══════════════════════════════════════════════════════════════
    # NODE 3: SEVERITY ASSESSMENT (set by nodes/severity.py)
    # ═══════════════════════════════════════════════════════════════
    severity: Optional[str] = None
    """Severity level: None (healthy), Mild, Moderate, Critical"""
    
    confidence: Optional[str] = None
    """Diagnosis confidence: High, Medium, Low"""
    
    is_healthy: bool = False
    """Flag indicating plant has no detected issues"""
    
    # ═══════════════════════════════════════════════════════════════
    # NODE 4: CAUSE ANALYSIS (set by nodes/causes.py)
    # ═══════════════════════════════════════════════════════════════
    causes: List[str] = []
    """Likely causes for detected symptoms (LLM-generated)"""
    
    # ═══════════════════════════════════════════════════════════════
    # NODE 5: SEASONAL CONTEXT (set by nodes/seasonal.py)
    # ═══════════════════════════════════════════════════════════════
    seasonal_insight: Optional[str] = None
    """Season-specific advice or warning"""
    
    # ═══════════════════════════════════════════════════════════════
    # NODE 6: CARE PLAN (set by nodes/care_plan.py)
    # ═══════════════════════════════════════════════════════════════
    care_immediate: List[str] = []
    """Urgent actions to take right now"""
    
    care_ongoing: List[str] = []
    """Long-term maintenance actions"""
    
    # ═══════════════════════════════════════════════════════════════
    # NODE 7: SAFETY FILTER (set by nodes/safety.py)
    # ═══════════════════════════════════════════════════════════════
    dont_do: List[str] = []
    """Actions to avoid (common mistakes)"""
    
    pro_tip: Optional[str] = None
    """Expert tip specific to this plant/situation"""
    
    # ═══════════════════════════════════════════════════════════════
    # NODE 8: RESPONSE FORMATTING (set by nodes/formatter.py)
    # ═══════════════════════════════════════════════════════════════
    rescan_suggested: bool = False
    """Flag to suggest user rescan with better conditions"""
    
    final_response: Optional[str] = None
    """Complete formatted markdown response for display"""


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
