"""
FloraVision AI - Node 7: Safety Filter
=======================================

PURPOSE:
    Adds "What NOT to do" warnings.
    Explicity lists harmful actions and common mistakes to avoid.
    Prefers organic, low-risk solutions.

POSITION IN PIPELINE:
    ┌─────────────────────────────────────────────────────────────────┐
    │  ... → Node 6 → [Node 7: Safety] → Node 8 → END                │
    │                       ▲                                         │
    │                       │                                         │
    │                   YOU ARE HERE                                  │
    └─────────────────────────────────────────────────────────────────┘

READS FROM STATE:
    - plant_name
    - symptoms_grouped
    - severity
    - is_healthy
    - care_immediate
    - care_ongoing

WRITES TO STATE:
    - dont_do (list of actions to avoid)
    - pro_tip (expert tip for this plant/situation)

CONNECTS TO:
    - Previous: nodes/care_plan.py (Node 6)
    - Next: nodes/formatter.py (Node 8)
    - Uses: knowledge/plants.json
"""

import json
import random
from pathlib import Path
from typing import List
from ..state import PlantState


# Load plants knowledge base
PLANTS_PATH = Path(__file__).parent.parent / "knowledge" / "plants.json"

with open(PLANTS_PATH) as f:
    PLANTS_DATA = json.load(f)


# Common mistakes by symptom category
DONT_DO_BY_CATEGORY = {
    "water": [
        "Don't water on a fixed schedule - always check soil moisture first",
        "Don't let the plant sit in standing water",
        "Don't use ice-cold water - room temperature is best"
    ],
    "nutrient": [
        "Don't over-fertilize - more is not better",
        "Don't fertilize a stressed or sick plant",
        "Don't use fertilizer at full strength - dilute to half"
    ],
    "fungal": [
        "Don't mist leaves if fungal issues are present",
        "Don't reuse soil from infected plants",
        "Don't place plants too close together - they need airflow"
    ],
    "pest": [
        "Don't use harsh chemical pesticides as first resort",
        "Don't ignore early signs - pests multiply quickly",
        "Don't forget to check the undersides of leaves"
    ],
    "light": [
        "Don't move plants to drastically different light suddenly",
        "Don't place plants in direct afternoon sun without acclimation",
        "Don't assume more light is always better"
    ],
    "stress": [
        "Don't repot a stressed plant - wait until it recovers",
        "Don't move the plant frequently",
        "Don't prune more than 1/3 of the plant at once"
    ],
    "disease": [
        "Don't overwater - it's the #1 cause of plant death",
        "Don't use containers without drainage holes",
        "Don't ignore foul smells from soil - it indicates rot"
    ]
}

GENERAL_DONT_DO = [
    "Don't panic - most plant issues are fixable with patience",
    "Don't overwater - it's more harmful than underwatering",
    "Don't place plants near heating or cooling vents",
    "Don't use tap water if your area has hard water - let it sit overnight",
    "Don't repot in a much larger pot - go up only 1-2 inches in diameter"
]


def safety_node(state: PlantState) -> dict:
    """
    Node 7: Safety Filter
    
    Adds warnings about what NOT to do.
    Also provides an expert pro tip specific to this situation.
    
    Args:
        state: Current PlantState
        
    Returns:
        dict with dont_do list and pro_tip
    """
    dont_do = _generate_dont_do(state)
    pro_tip = _generate_pro_tip(state)
    
    # Add reasoning trace
    trace = f"Safety: Added {len(dont_do)} warnings and 1 pro tip."
    
    return {
        "dont_do": dont_do,
        "pro_tip": pro_tip,
        "reasoning_trace": state.reasoning_trace + [trace]
    }


def _generate_dont_do(state: PlantState) -> List[str]:
    """
    Generate "what not to do" warnings based on symptoms.
    """
    warnings = []
    
    # Add symptom-specific warnings
    for category in state.symptoms_grouped.keys():
        if category in DONT_DO_BY_CATEGORY:
            # Pick 1-2 relevant warnings from this category
            category_warnings = DONT_DO_BY_CATEGORY[category]
            warnings.extend(random.sample(
                category_warnings, 
                min(1, len(category_warnings))
            ))
    
    # Add general warnings if we don't have enough
    if len(warnings) < 2:
        # Add from general list
        available = [w for w in GENERAL_DONT_DO if w not in warnings]
        needed = 2 - len(warnings)
        warnings.extend(random.sample(available, min(needed, len(available))))
    
    # For healthy plants, different advice
    if state.is_healthy:
        warnings = [
            "Don't overwater just because you want to 'help' - let soil dry between waterings",
            "Don't move a thriving plant - if it's happy, leave it be"
        ]
    
    return warnings[:3]  # Max 3 warnings


def _generate_pro_tip(state: PlantState) -> str:
    """
    Generate a pro tip specific to this plant/situation.
    """
    plant_info = PLANTS_DATA.get(state.plant_name, PLANTS_DATA.get("unknown", {}))
    pro_tips = plant_info.get("pro_tips", [])
    
    if pro_tips:
        # Pick a random pro tip from the plant's knowledge base
        return random.choice(pro_tips)
    
    # Fallback tips based on severity
    if state.is_healthy:
        return "Healthy plants can be propagated! Consider taking cuttings to share with friends. 🌱"
    
    if state.severity == "Critical":
        return "Don't give up! Even severely damaged plants can recover with proper care and patience."
    
    return "Observe your plant daily - you'll learn to read its signals over time."
