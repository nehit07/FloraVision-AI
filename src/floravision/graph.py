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
    """
    Create the LangGraph reasoning pipeline.
    
    This builds the graph but does NOT compile it yet.
    Call .compile() on the result to get a runnable graph.
    
    Returns:
        StateGraph with all 8 nodes wired together
    
    Example:
        graph = create_graph()
        compiled = graph.compile()
        result = compiled.invoke(initial_state)
    """
    # Initialize StateGraph with our state schema
    graph = StateGraph(PlantState)
    
    # ═══════════════════════════════════════════════════════════════
    # ADD ALL 8 NODES
    # ═══════════════════════════════════════════════════════════════
    
    graph.add_node("identification", identification_node)   # Node 1
    graph.add_node("symptoms", symptoms_node)               # Node 2
    graph.add_node("severity", severity_node)               # Node 3
    graph.add_node("causes", causes_node)                   # Node 4
    graph.add_node("seasonal", seasonal_node)               # Node 5
    graph.add_node("care_plan", care_plan_node)             # Node 6
    graph.add_node("safety", safety_node)                   # Node 7
    graph.add_node("formatter", formatter_node)             # Node 8
    
    # ═══════════════════════════════════════════════════════════════
    # DEFINE EDGES (LINEAR FLOW)
    # ═══════════════════════════════════════════════════════════════
    
    # Set entry point
    graph.set_entry_point("identification")
    
    # Wire nodes in sequence
    graph.add_edge("identification", "symptoms")
    graph.add_edge("symptoms", "severity")
    graph.add_edge("severity", "causes")
    graph.add_edge("causes", "seasonal")
    graph.add_edge("seasonal", "care_plan")
    graph.add_edge("care_plan", "safety")
    graph.add_edge("safety", "formatter")
    
    # Set exit point
    graph.add_edge("formatter", END)
    
    return graph


def get_compiled_graph():
    """
    Get a compiled, ready-to-run graph.
    
    Returns:
        Compiled LangGraph ready for .invoke()
    """
    return create_graph().compile()


def run_diagnosis(
    image_bytes: bytes,
    season: str = "unknown",
    mock: bool = True
) -> str:
    """
    Complete diagnosis pipeline - from image to formatted response.
    
    This is the main entry point for running a diagnosis.
    It handles:
    1. YOLO detection (populates yolo_detections)
    2. Plant identification (populates plant_name, plant_id_confidence)
    3. Running the 8-node reasoning pipeline
    4. Returning the formatted response
    
    Args:
        image_bytes: Raw image bytes from camera/upload
        season: Current season (spring/summer/autumn/winter)
        mock: Use mock mode for detection (demo purposes)
        
    Returns:
        Formatted markdown diagnosis string
        
    Example:
        with open("plant.jpg", "rb") as f:
            image = f.read()
        
        diagnosis = run_diagnosis(image, season="winter", mock=True)
        print(diagnosis)
    """
    # Step 1: Run YOLO detection
    yolo_detections = detect_symptoms(image_bytes, mock=mock)
    
    # Step 2: Identify plant species
    plant_name, plant_confidence = identify_plant(image_bytes, mock=mock)
    
    # Step 3: Create initial state
    initial_state = PlantState(
        image=image_bytes,
        season=season.lower(),
        plant_name=plant_name,
        plant_id_confidence=plant_confidence,
        yolo_detections=yolo_detections
    )
    
    # Step 4: Run the reasoning pipeline
    compiled_graph = get_compiled_graph()
    final_state = compiled_graph.invoke(initial_state)
    
    # Step 5: Return the formatted response
    return final_state["final_response"]


def run_diagnosis_full(
    image_bytes: bytes,
    season: str = "unknown",
    mock: bool = True
) -> PlantState:
    """
    Run diagnosis and return the complete state (for debugging/advanced use).
    
    Args:
        image_bytes: Raw image bytes
        season: Current season
        mock: Use mock mode
        
    Returns:
        Complete PlantState with all fields populated
    """
    # Run YOLO and plant ID
    yolo_detections = detect_symptoms(image_bytes, mock=mock)
    plant_name, plant_confidence = identify_plant(image_bytes, mock=mock)
    
    # Create and run
    initial_state = PlantState(
        image=image_bytes,
        season=season.lower(),
        plant_name=plant_name,
        plant_id_confidence=plant_confidence,
        yolo_detections=yolo_detections
    )
    
    compiled_graph = get_compiled_graph()
    return compiled_graph.invoke(initial_state)
