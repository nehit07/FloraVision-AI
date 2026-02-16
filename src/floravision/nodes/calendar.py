"""
FloraVision AI - Node 9: Care Calendar Generation
=================================================

PURPOSE:
    Transforms care recommendations into a weekly schedule.
"""

from typing import List, Dict
from ..state import PlantState

def calendar_node(state: PlantState) -> dict:
    """
    Node 9: Care Calendar Generation
    
    Args:
        state: Current PlantState
        
    Returns:
        dict with care_calendar list
    """
    calendar = _generate_calendar(state)
    
    trace = f"Calendar: Generated a {len(calendar)}-day care schedule."
    
    return {
        "care_calendar": calendar,
        "reasoning_trace": state.reasoning_trace + [trace]
    }

def _generate_calendar(state: PlantState) -> List[Dict[str, str]]:
    """
    Distribute care tasks across the week.
    """
    calendar = []
    
    # Healthy plants have a simple maintenance schedule
    if state.is_healthy:
        calendar = [
            {"day": "Monday", "task": "Check soil moisture & water if dry"},
            {"day": "Wednesday", "task": "Rotate plant for even light"},
            {"day": "Friday", "task": "Inspect leaves for dust or pests"}
        ]
    else:
        # Map ongoing tasks to specific days
        tasks = state.care_ongoing
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        # Priority mapping
        for i, task in enumerate(tasks):
            if i < len(days):
                calendar.append({"day": days[i*2 % 7], "task": task})
        
        # Ensure at least one task if ongoing is empty
        if not calendar:
            calendar.append({"day": "Daily", "task": "Monitor plant recovery progress"})
            
    return calendar
