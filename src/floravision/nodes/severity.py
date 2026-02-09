"""
FloraVision AI - Node 3: Severity Assessment
=============================================

PURPOSE:
    Classifies plant condition severity using RULE-BASED logic.
    This is DETERMINISTIC, not LLM-generated, for reliability.

POSITION IN PIPELINE:
    ┌─────────────────────────────────────────────────────────────────┐
    │  Node 1 → Node 2 → [Node 3: Severity] → Node 4 → ... → END     │
    │                          ▲                                      │
    │                          │                                      │
    │                      YOU ARE HERE                               │
    └─────────────────────────────────────────────────────────────────┘

READS FROM STATE:
    - yolo_detections (for confidence calculation)
    - symptoms_grouped (from Node 2)
    - plant_id_confidence (for overall confidence)

WRITES TO STATE:
    - severity (None/Mild/Moderate/Critical)
    - confidence (High/Medium/Low)
    - is_healthy (boolean)

SEVERITY RULES (Rule-Based, NOT LLM):
    - total_weight >= 6 OR fungal detected → Critical
    - total_weight >= 3 → Moderate
    - total_weight >= 1 → Mild
    - total_weight == 0 → Healthy (is_healthy = True)

CONNECTS TO:
    - Previous: nodes/symptoms.py (Node 2)
    - Next: nodes/causes.py (Node 4)
    - Uses: knowledge/symptoms.json (for severity weights)
"""

import json
from pathlib import Path
from ..state import PlantState, calculate_confidence


# Load symptoms for severity weights
SYMPTOMS_PATH = Path(__file__).parent.parent / "knowledge" / "symptoms.json"

with open(SYMPTOMS_PATH) as f:
    SYMPTOMS_DATA = json.load(f)


def severity_node(state: PlantState) -> dict:
    """
    Node 3: Severity Assessment
    
    Uses RULE-BASED logic to determine severity.
    This ensures consistent, predictable severity ratings.
    
    Severity Levels:
        - None: Plant is healthy, no issues detected
        - Mild: Minor issues, easy to fix
        - Moderate: Needs attention soon
        - Critical: Urgent intervention required
    
    Args:
        state: Current PlantState
        
    Returns:
        dict with severity, confidence, is_healthy
    """
    updates = {}
    
    # Calculate total severity weight
    total_weight = 0
    has_fungal = False
    has_disease = False
    
    for detection in state.yolo_detections:
        label = detection.label
        
        if label in SYMPTOMS_DATA:
            symptom_info = SYMPTOMS_DATA[label]
            
            # Add severity weight (scaled by detection confidence)
            weight = symptom_info.get("severity_weight", 1)
            total_weight += weight * detection.confidence
            
            # Check for high-risk categories
            category = symptom_info.get("category", "")
            if category == "fungal":
                has_fungal = True
            if category == "disease":
                has_disease = True
    
    # ═══════════════════════════════════════════════════════════════
    # RULE-BASED SEVERITY DETERMINATION
    # ═══════════════════════════════════════════════════════════════
    
    if total_weight == 0 and len(state.yolo_detections) == 0:
        # No symptoms detected - healthy plant!
        updates["severity"] = None
        updates["is_healthy"] = True
    
    elif total_weight >= 6 or has_fungal or has_disease:
        # Critical: High weight OR fungal/disease detected
        updates["severity"] = "Critical"
        updates["is_healthy"] = False
    
    elif total_weight >= 3:
        # Moderate: Needs attention
        updates["severity"] = "Moderate"
        updates["is_healthy"] = False
    
    elif total_weight >= 1:
        # Mild: Minor issues
        updates["severity"] = "Mild"
        updates["is_healthy"] = False
    
    else:
        # Edge case: Very low weight symptoms
        updates["severity"] = "Mild"
        updates["is_healthy"] = False
    
    # ═══════════════════════════════════════════════════════════════
    # CONFIDENCE CALCULATION
    # ═══════════════════════════════════════════════════════════════
    
    updates["confidence"] = calculate_confidence(state)
    
    return updates


def get_severity_description(severity: str | None) -> str:
    """
    Get human-readable description for severity level.
    
    Args:
        severity: Severity level or None
        
    Returns:
        Description string
    """
    descriptions = {
        None: "Your plant looks healthy! No issues detected.",
        "Mild": "Minor issues detected. Easy to address with simple care adjustments.",
        "Moderate": "Your plant needs attention. Address these issues soon to prevent worsening.",
        "Critical": "Urgent care needed! Take action immediately to save your plant."
    }
    return descriptions.get(severity, "Unable to determine severity.")
