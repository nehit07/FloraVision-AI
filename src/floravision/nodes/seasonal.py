"""
FloraVision AI - Node 5: Seasonal Context Adjustment
=====================================================

PURPOSE:
    Adjusts care recommendations based on current season.
    Adds season-specific warnings and insights.

POSITION IN PIPELINE:
    ┌─────────────────────────────────────────────────────────────────┐
    │  ... → Node 4 → [Node 5: Seasonal] → Node 6 → Node 7 → END     │
    │                        ▲                                        │
    │                        │                                        │
    │                    YOU ARE HERE                                 │
    └─────────────────────────────────────────────────────────────────┘

READS FROM STATE:
    - season
    - symptoms_grouped
    - is_healthy

WRITES TO STATE:
    - seasonal_insight (season-specific advice)

CONNECTS TO:
    - Previous: nodes/causes.py (Node 4)
    - Next: nodes/care_plan.py (Node 6)
    - Uses: knowledge/seasons.json

SEASONAL CONSIDERATIONS:
    - Winter: Reduced watering, no fertilizing, dormancy normal
    - Spring: Growth starting, good time to repot
    - Summer: More water, watch for pests
    - Autumn: Reduce care, prepare for dormancy
"""

import json
from pathlib import Path
from ..state import PlantState


# Load seasons knowledge base
SEASONS_PATH = Path(__file__).parent.parent / "knowledge" / "seasons.json"

with open(SEASONS_PATH) as f:
    SEASONS_DATA = json.load(f)


def seasonal_node(state: PlantState) -> dict:
    """
    Node 5: Seasonal Context Adjustment
    
    Provides season-specific insights and warnings.
    Helps user understand how current season affects plant care.
    
    Args:
        state: Current PlantState
        
    Returns:
        dict with seasonal_insight
    """
    season = state.season.lower()
    
    # Get season data (fallback to unknown if invalid)
    if season not in SEASONS_DATA:
        season = "unknown"
    
    season_info = SEASONS_DATA[season]
    
    # Build seasonal insight
    insight = _build_seasonal_insight(state, season, season_info)
    
    return {"seasonal_insight": insight}


def _build_seasonal_insight(
    state: PlantState, 
    season: str, 
    season_info: dict
) -> str:
    """
    Build a personalized seasonal insight message.
    """
    insights = []
    
    # Start with season description
    if season != "unknown":
        insights.append(f"**{season.title()}**: {season_info['description']}.")
    
    # Add relevant warnings based on symptoms
    warnings = season_info.get("warnings", [])
    symptoms = state.symptoms_grouped
    
    # Match warnings to symptoms
    if "water" in symptoms and season == "winter":
        insights.append("⚠️ Overwatering is especially dangerous in winter when plants are dormant.")
    
    if "fungal" in symptoms and season in ["summer", "autumn"]:
        insights.append("⚠️ Humid conditions in this season can worsen fungal issues.")
    
    if "light" in symptoms and season == "winter":
        insights.append("💡 Shorter days mean less light - consider moving your plant closer to windows.")
    
    # Add general seasonal advice
    if state.is_healthy:
        advice = season_info.get("general_advice", [])
        if advice:
            insights.append(f"Tip: {advice[0]}")
    else:
        # Add first relevant warning
        if warnings:
            insights.append(f"Note: {warnings[0]}")
    
    # Add watering modifier info
    modifier = season_info.get("watering_modifier", 1.0)
    if modifier < 0.8:
        insights.append("Reduce watering frequency during this season.")
    elif modifier > 1.2:
        insights.append("Your plant may need more frequent watering in this season.")
    
    return " ".join(insights)


def get_season_from_month(month: int) -> str:
    """
    Determine season from month (Northern Hemisphere).
    
    Args:
        month: Month number (1-12)
        
    Returns:
        Season name
    """
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"


def get_watering_modifier(season: str) -> float:
    """
    Get watering frequency modifier for a season.
    
    Args:
        season: Season name
        
    Returns:
        Modifier (1.0 = normal, <1 = less, >1 = more)
    """
    if season in SEASONS_DATA:
        return SEASONS_DATA[season].get("watering_modifier", 1.0)
    return 1.0
