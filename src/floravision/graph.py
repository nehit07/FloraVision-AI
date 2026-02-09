"""
FloraVision AI - LangGraph Reasoning Pipeline
==============================================

PURPOSE:
    Wires together all 8 reasoning nodes into a LangGraph StateGraph.
    This is the MAIN ORCHESTRATION FILE for the diagnosis pipeline.

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     LangGraph Pipeline                          │
    │                                                                 │
    │  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐       │
    │  │ Node 1  │──▶│  Node 2  │──▶│  Node 3  │──▶│ Node 4  │       │
    │  │ Plant ID│   │ Symptoms │   │ Severity │   │ Causes  │       │
    │  └─────────┘   └──────────┘   └──────────┘   └─────────┘       │
    │       │                                           │             │
    │       │                                           ▼             │
    │  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐       │
    │  │ Node 8  │◀──│  Node 7  │◀──│  Node 6  │◀──│ Node 5  │       │
    │  │Formatter│   │  Safety  │   │Care Plan │   │Seasonal │       │
    │  └─────────┘   └──────────┘   └──────────┘   └─────────┘       │
    │       │                                                         │
    │       ▼                                                         │
    │   [END] ──▶ final_response ready for display                   │
    └─────────────────────────────────────────────────────────────────┘

CONNECTIONS:
    - Imported by: app.py (Streamlit UI)
    - Uses: All 8 nodes from nodes/
    - Uses: detection/ modules for preprocessing
    - Uses: state.py for PlantState

USAGE:
    from floravision.graph import create_graph, run_diagnosis
    
    # Option 1: Create graph and invoke manually
    graph = create_graph()
    result = graph.invoke(initial_state)
    
    # Option 2: Use convenience function
    response = run_diagnosis(image_bytes, season="winter")
"""

from langgraph.graph import StateGraph, END
from .state import PlantState

# Import all 8 reasoning nodes
from .nodes.identification import identification_node
from .nodes.symptoms import symptoms_node
from .nodes.severity import severity_node
from .nodes.causes import causes_node
from .nodes.seasonal import seasonal_node
from .nodes.care_plan import care_plan_node
from .nodes.safety import safety_node
from .nodes.formatter import formatter_node

# Import detection modules
from .detection.yolo_detector import detect_symptoms
from .detection.plant_id import identify_plant


def create_graph() -> StateGraph:
    graph = StateGraph(PlantState)
    
    graph.add_node("identification", identification_node)   # Node 1
    graph.add_node("symptoms", symptoms_node)               # Node 2
    graph.add_node("severity", severity_node)               # Node 3
    graph.add_node("causes", causes_node)                   # Node 4
    graph.add_node("seasonal", seasonal_node)               # Node 5
    graph.add_node("care_plan", care_plan_node)             # Node 6
    graph.add_node("safety", safety_node)                   # Node 7
    graph.add_node("formatter", formatter_node)             # Node 8
    
    # Set entry point
    graph.set_entry_point("identification")
    
    # Normal flow: identification → symptoms → severity
    graph.add_edge("identification", "symptoms")
    graph.add_edge("symptoms", "severity")
    
    # CONDITIONAL: After severity, route based on health status
    # Healthy plants skip cause analysis, go directly to formatter
    graph.add_conditional_edges(
        "severity",
        _route_after_severity,
        {
            "healthy": "formatter",    # Skip causes/care/safety for healthy plants
            "needs_care": "causes"     # Continue normal flow for symptomatic plants
        }
    )
    
    # Normal flow continuation for symptomatic plants
    graph.add_edge("causes", "seasonal")
    graph.add_edge("seasonal", "care_plan")
    graph.add_edge("care_plan", "safety")
    graph.add_edge("safety", "formatter")
    
    # Set exit point
    graph.add_edge("formatter", END)
    
    return graph


def _route_after_severity(state: PlantState) -> str:
    if state.is_healthy:
        return "healthy"
    return "needs_care"



def get_compiled_graph():
    return create_graph().compile()


def run_diagnosis(
    image_bytes: bytes,
    season: str = "unknown",
    mock: bool = True
) -> str:
    yolo_detections = detect_symptoms(image_bytes, mock=mock)
    plant_name, plant_confidence = identify_plant(image_bytes, mock=mock)
    
    initial_state = PlantState(
        image=image_bytes,
        season=season.lower(),
        plant_name=plant_name,
        plant_id_confidence=plant_confidence,
        yolo_detections=yolo_detections
    )
    
    compiled_graph = get_compiled_graph()
    final_state = compiled_graph.invoke(initial_state)
    return final_state["final_response"]


def run_diagnosis_full(
    image_bytes: bytes,
    season: str = "unknown",
    mock: bool = True
) -> PlantState:

    yolo_detections = detect_symptoms(image_bytes, mock=mock)
    plant_name, plant_confidence = identify_plant(image_bytes, mock=mock)
    
    initial_state = PlantState(
        image=image_bytes,
        season=season.lower(),
        plant_name=plant_name,
        plant_id_confidence=plant_confidence,
        yolo_detections=yolo_detections
    )
    
    compiled_graph = get_compiled_graph()
    result = compiled_graph.invoke(initial_state)
    return PlantState(**result)
