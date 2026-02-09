"""
FloraVision AI - Node 1: Plant Identification
==============================================

PURPOSE:
    Confirms the plant species identification from the detection layer.
    Applies confidence threshold and sets unknown mode if needed.

POSITION IN PIPELINE:
    ┌─────────────────────────────────────────────────────────────────┐
    │  START → [Node 1: Identification] → Node 2 → ... → END         │
    │              ▲                                                  │
    │              │                                                  │
    │          YOU ARE HERE                                           │
    └─────────────────────────────────────────────────────────────────┘

READS FROM STATE:
    - plant_name (from detection/plant_id.py)
    - plant_id_confidence (from detection/plant_id.py)

WRITES TO STATE:
    - plant_name (may update to "Unknown" if low confidence)

CONNECTS TO:
    - Previous: Called first in graph.py pipeline
    - Next: nodes/symptoms.py (Node 2)
"""

import json
from pathlib import Path
from ..state import PlantState


# Load plants knowledge base
PLANTS_PATH = Path(__file__).parent.parent / "knowledge" / "plants.json"

with open(PLANTS_PATH) as f:
    PLANTS_DATA = json.load(f)


# Confidence threshold - below this, mark as Unknown
CONFIDENCE_THRESHOLD = 0.6


def identification_node(state: PlantState) -> dict:
    """
    Node 1: Plant Identification
    
    Validates the plant identification and applies confidence rules.
    
    Rules:
        - If confidence < 0.6 → plant_name = "unknown"
        - If plant not in database → plant_name = "unknown"
        - Otherwise, keep the identified plant name
    
    Args:
        state: Current PlantState
        
    Returns:
        dict with updated plant_name (if needed)
    """
    updates = {}
    
    # Get current identification
    plant_name = state.plant_name.lower()
    confidence = state.plant_id_confidence
    
    # Rule 1: Check confidence threshold
    if confidence < CONFIDENCE_THRESHOLD:
        updates["plant_name"] = "unknown"
        return updates
    
    # Rule 2: Check if plant is in our knowledge base
    if plant_name not in PLANTS_DATA:
        updates["plant_name"] = "unknown"
        return updates
    
    # Plant is valid and confident - normalize the name
    updates["plant_name"] = plant_name
    
    return updates
