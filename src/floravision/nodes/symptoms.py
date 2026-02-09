"""
FloraVision AI - Node 2: Symptom Interpretation
================================================

PURPOSE:
    Maps YOLO detection labels to plant stress categories.
    Groups symptoms for easier analysis by downstream nodes.

POSITION IN PIPELINE:
    ┌─────────────────────────────────────────────────────────────────┐
    │  Node 1 → [Node 2: Symptoms] → Node 3 → ... → END              │
    │              ▲                                                  │
    │              │                                                  │
    │          YOU ARE HERE                                           │
    └─────────────────────────────────────────────────────────────────┘

READS FROM STATE:
    - yolo_detections (from detection/yolo_detector.py)

WRITES TO STATE:
    - symptoms_grouped (dict with categories as keys)

CONNECTS TO:
    - Previous: nodes/identification.py (Node 1)
    - Next: nodes/severity.py (Node 3)
    - Uses: knowledge/symptoms.json (for category mapping)

CATEGORIES:
    - nutrient: Nitrogen, iron, other nutrient deficiencies
    - water: Overwatering, underwatering, humidity issues
    - fungal: Fungal diseases like powdery mildew
    - pest: Insect damage
    - light: Too much or too little light
    - stress: General environmental stress
    - disease: Root rot and other diseases
"""

import json
from pathlib import Path
from typing import Dict, List
from ..state import PlantState


# Load symptoms knowledge base
SYMPTOMS_PATH = Path(__file__).parent.parent / "knowledge" / "symptoms.json"

with open(SYMPTOMS_PATH) as f:
    SYMPTOMS_DATA = json.load(f)


def symptoms_node(state: PlantState) -> dict:
    """
    Node 2: Symptom Interpretation
    
    Maps YOLO labels to stress categories and groups them.
    
    Example output:
        {
            "nutrient": ["leaf_yellowing", "pale_leaves"],
            "fungal": ["brown_spots"],
            "water": ["wilting"]
        }
    
    Args:
        state: Current PlantState with yolo_detections
        
    Returns:
        dict with symptoms_grouped
    """
    # Initialize category groups
    grouped: Dict[str, List[str]] = {}
    
    # Process each detection
    for detection in state.yolo_detections:
        label = detection.label
        
        # Look up symptom in knowledge base
        if label in SYMPTOMS_DATA:
            symptom_info = SYMPTOMS_DATA[label]
            category = symptom_info["category"]
            
            # Add to appropriate category group
            if category not in grouped:
                grouped[category] = []
            
            if label not in grouped[category]:
                grouped[category].append(label)
        else:
            # Unknown symptom - add to general "stress" category
            if "stress" not in grouped:
                grouped["stress"] = []
            grouped["stress"].append(label)
    
    # Add reasoning trace
    if grouped:
        categories = list(grouped.keys())
        symptom_count = sum(len(v) for v in grouped.values())
        trace = f"Symptoms: Grouped {symptom_count} symptom(s) into {len(categories)} categories: {', '.join(categories)}."
    else:
        trace = "Symptoms: No symptoms detected."
    
    return {
        "symptoms_grouped": grouped,
        "reasoning_trace": state.reasoning_trace + [trace]
    }


def get_symptom_display_name(label: str) -> str:
    """
    Get human-readable name for a symptom label.
    
    Args:
        label: Raw symptom label (e.g., "leaf_yellowing")
        
    Returns:
        Display name (e.g., "Leaf Yellowing")
    """
    if label in SYMPTOMS_DATA:
        return SYMPTOMS_DATA[label].get("display_name", label.replace("_", " ").title())
    return label.replace("_", " ").title()


def get_symptom_causes(label: str) -> List[str]:
    """
    Get possible causes for a symptom.
    
    Args:
        label: Symptom label
        
    Returns:
        List of possible causes
    """
    if label in SYMPTOMS_DATA:
        return SYMPTOMS_DATA[label].get("possible_causes", [])
    return []
