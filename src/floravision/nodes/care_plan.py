"""
FloraVision AI - Node 6: Care Plan Generation
==============================================

PURPOSE:
    Generates actionable care recommendations.
    Separates immediate actions from ongoing maintenance.

POSITION IN PIPELINE:
    ┌─────────────────────────────────────────────────────────────────┐
    │  ... → Node 5 → [Node 6: Care Plan] → Node 7 → Node 8 → END    │
    │                        ▲                                        │
    │                        │                                        │
    │                    YOU ARE HERE                                 │
    └─────────────────────────────────────────────────────────────────┘

READS FROM STATE:
    - plant_name
    - symptoms_grouped
    - severity
    - is_healthy
    - causes

WRITES TO STATE:
    - care_immediate (what to do NOW)
    - care_ongoing (maintenance actions)

CONNECTS TO:
    - Previous: nodes/seasonal.py (Node 5)
    - Next: nodes/safety.py (Node 7)
    - Uses: knowledge/plants.json, knowledge/symptoms.json
"""

import json
from pathlib import Path
from typing import List, Tuple
from ..state import PlantState


# Load knowledge bases
PLANTS_PATH = Path(__file__).parent.parent / "knowledge" / "plants.json"
SYMPTOMS_PATH = Path(__file__).parent.parent / "knowledge" / "symptoms.json"

with open(PLANTS_PATH) as f:
    PLANTS_DATA = json.load(f)

with open(SYMPTOMS_PATH) as f:
    SYMPTOMS_DATA = json.load(f)


def care_plan_node(state: PlantState) -> dict:
    """
    Node 6: Care Plan Generation
    
    Creates two types of care actions:
    - Immediate: What to do RIGHT NOW
    - Ongoing: Maintenance for the coming weeks
    
    Args:
        state: Current PlantState
        
    Returns:
        dict with care_immediate and care_ongoing lists
    """
    immediate, ongoing = _generate_care_plan(state)
    
    # Add reasoning trace
    trace = f"Care Plan: Generated {len(immediate)} immediate and {len(ongoing)} ongoing actions."
    
    return {
        "care_immediate": immediate,
        "care_ongoing": ongoing,
        "reasoning_trace": state.reasoning_trace + [trace]
    }


def _generate_care_plan(state: PlantState) -> Tuple[List[str], List[str]]:
    """
    Generate care recommendations based on symptoms and plant type.
    """
    immediate = []
    ongoing = []
    
    # Get plant-specific info
    plant_info = PLANTS_DATA.get(state.plant_name, PLANTS_DATA.get("unknown", {}))
    
    # ═══════════════════════════════════════════════════════════════
    # HEALTHY PLANT PATH
    # ═══════════════════════════════════════════════════════════════
    if state.is_healthy:
        immediate = [
            "Your plant looks great! No immediate action needed.",
            "Take a moment to appreciate your healthy plant! 🌿"
        ]
        ongoing = [
            f"Water {plant_info.get('water_frequency', 'when soil is dry')}",
            f"Provide {plant_info.get('light', 'appropriate')} light",
            "Check for pests weekly during your watering routine"
        ]
        return immediate, ongoing
    
    # ═══════════════════════════════════════════════════════════════
    # SYMPTOM-BASED CARE RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════
    
    symptoms = state.symptoms_grouped
    
    # WATER ISSUES
    if "water" in symptoms:
        water_symptoms = symptoms["water"]
        if "wilting" in water_symptoms:
            immediate.append("Check soil moisture immediately - if dry, water thoroughly until it drains")
            immediate.append("Move plant to a cooler, shadier spot temporarily")
        if "brown_tips" in water_symptoms:
            immediate.append("Mist leaves or place a humidity tray nearby")
            ongoing.append("Increase humidity around the plant")
    
    # NUTRIENT ISSUES
    if "nutrient" in symptoms:
        immediate.append("Check if plant is root-bound - look for roots circling the pot")
        ongoing.append("Apply a balanced liquid fertilizer (diluted to half strength)")
        ongoing.append("Consider repotting if roots are crowded")
    
    # FUNGAL ISSUES
    if "fungal" in symptoms:
        immediate.append("Isolate this plant from others to prevent spread")
        immediate.append("Remove affected leaves with clean scissors")
        ongoing.append("Improve air circulation around the plant")
        ongoing.append("Avoid getting water on leaves when watering")
    
    # PEST ISSUES
    if "pest" in symptoms:
        immediate.append("Inspect all leaves (top and bottom) for pests")
        immediate.append("Wipe leaves with diluted neem oil or soapy water")
        ongoing.append("Check nearby plants for pest spread")
        ongoing.append("Repeat neem treatment weekly for 3 weeks")
    
    # LIGHT ISSUES
    if "light" in symptoms:
        light_symptoms = symptoms["light"]
        if "pale_leaves" in light_symptoms or "leggy_growth" in light_symptoms:
            immediate.append("Move plant to a brighter location")
            ongoing.append("Rotate plant weekly for even growth")
        else:
            immediate.append("Move plant away from direct sunlight")
    
    # DISEASE ISSUES
    if "disease" in symptoms:
        if "root_rot" in symptoms.get("disease", []):
            immediate.append("Remove plant from pot and inspect roots")
            immediate.append("Cut away any brown, mushy roots with sterile scissors")
            immediate.append("Repot in fresh, well-draining soil")
            ongoing.append("Reduce watering frequency significantly")
    
    # GENERAL STRESS
    if "stress" in symptoms:
        immediate.append("Avoid making major changes - let plant stabilize")
        ongoing.append("Maintain consistent watering schedule")
        ongoing.append("Keep plant in a stable environment")
    
    # ═══════════════════════════════════════════════════════════════
    # ADD PLANT-SPECIFIC RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════
    
    # Add general ongoing care
    water_freq = plant_info.get("water_frequency", "when top inch of soil is dry")
    if not any("water" in item.lower() for item in ongoing):
        ongoing.append(f"Water {water_freq}")
    
    # Ensure we have at least some recommendations
    if not immediate:
        immediate.append("Monitor your plant closely over the next few days")
    
    if not ongoing:
        ongoing.append("Maintain regular watering and observation")
    
    return immediate[:4], ongoing[:4]  # Limit to 4 each
