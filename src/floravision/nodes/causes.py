"""
FloraVision AI - Node 4: Cause Analysis
========================================

PURPOSE:
    Uses LLM to explain the biological causes of detected symptoms.
    Provides simple, jargon-free explanations.

POSITION IN PIPELINE:
    ┌─────────────────────────────────────────────────────────────────┐
    │  Node 1 → Node 2 → Node 3 → [Node 4: Causes] → Node 5 → END    │
    │                                  ▲                              │
    │                                  │                              │
    │                              YOU ARE HERE                       │
    └─────────────────────────────────────────────────────────────────┘

READS FROM STATE:
    - plant_name
    - symptoms_grouped (from Node 2)
    - yolo_detections

WRITES TO STATE:
    - causes (list of cause explanations)

CONNECTS TO:
    - Previous: nodes/severity.py (Node 3)
    - Next: nodes/seasonal.py (Node 5)
    - Uses: LLM (Gemini) for reasoning
    - Fallback: knowledge/symptoms.json if LLM unavailable
"""

import os
import json
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from ..state import PlantState

# Load environment
load_dotenv()

# Load symptoms for fallback causes
SYMPTOMS_PATH = Path(__file__).parent.parent / "knowledge" / "symptoms.json"

with open(SYMPTOMS_PATH) as f:
    SYMPTOMS_DATA = json.load(f)


def causes_node(state: PlantState) -> dict:
    """
    Node 4: Cause Analysis
    
    Explains why the detected symptoms are occurring.
    Uses LLM for intelligent reasoning, falls back to knowledge base.
    
    Args:
        state: Current PlantState
        
    Returns:
        dict with causes list
    """
    # If plant is healthy, no causes needed
    if state.is_healthy:
        trace = "Causes: Plant is healthy, no cause analysis needed."
        return {
            "causes": [],
            "reasoning_trace": state.reasoning_trace + [trace]
        }
    
    # Try LLM-based cause analysis
    try:
        causes = _llm_cause_analysis(state)
        if causes:
            trace = f"Causes: LLM identified {len(causes)} cause(s)."
            return {
                "causes": causes,
                "reasoning_trace": state.reasoning_trace + [trace]
            }
    except Exception as e:
        print(f"LLM cause analysis failed: {e}")
    
    # Fallback to knowledge base
    causes = _knowledge_base_causes(state)
    trace = f"Causes: Knowledge base provided {len(causes)} cause(s) (LLM fallback)."
    return {
        "causes": causes,
        "reasoning_trace": state.reasoning_trace + [trace]
    }


def _llm_cause_analysis(state: PlantState) -> List[str]:
    """
    Use abstracted LLM to analyze causes.
    """
    from ..llm import get_llm
    llm = get_llm()
    
    if not llm.is_available:
        return []
    
    # Build symptom summary
    symptom_list = []
    for detection in state.yolo_detections:
        if detection.label in SYMPTOMS_DATA:
            display = SYMPTOMS_DATA[detection.label].get("display_name", detection.label)
            symptom_list.append(display)
    
    if not symptom_list:
        return []
    
    prompt = f"""You are a plant health expert. Analyze these symptoms for a {state.plant_name} plant.

Detected symptoms: {', '.join(symptom_list)}

Provide 2-3 likely causes in simple, friendly language. 
Each cause should be ONE short sentence.
Avoid technical jargon.
Focus on practical, actionable insights.

Format: Return only a numbered list, nothing else.
Example:
1. Overwatering is causing the roots to suffocate.
2. Low humidity is drying out the leaf tips.
"""
    
    response_text = llm.generate(prompt)
    if not response_text:
        return []
    
    # Parse numbered list
    causes = []
    for line in response_text.strip().split('\n'):
        line = line.strip()
        if line and line[0].isdigit():
            # Remove numbering
            cause = line.lstrip('0123456789.').strip()
            if cause:
                causes.append(cause)
    
    return causes[:3]  # Max 3 causes


def _knowledge_base_causes(state: PlantState) -> List[str]:
    """
    Fallback: Get causes from knowledge base.
    """
    causes = []
    seen = set()
    
    for detection in state.yolo_detections:
        if detection.label in SYMPTOMS_DATA:
            possible = SYMPTOMS_DATA[detection.label].get("possible_causes", [])
            for cause in possible:
                if cause not in seen:
                    causes.append(cause)
                    seen.add(cause)
    
    # Format causes as sentences
    formatted = []
    for cause in causes[:3]:
        # Convert snake_case to readable
        readable = cause.replace("_", " ").capitalize()
        formatted.append(f"{readable} may be affecting your plant.")
    
    return formatted
