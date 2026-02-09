# FloraVision AI — Project Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Technical Stack](#technical-stack)
3. [LangGraph Pipeline](#langgraph-pipeline)
4. [State Management](#state-management)
5. [Knowledge Base](#knowledge-base)
6. [API Reference](#api-reference)

---

## System Overview

FloraVision AI is a **multi-modal plant health assistant** that combines:
- **Computer Vision** (YOLO v8) for symptom detection
- **LLM Reasoning** (Gemini) for causal analysis
- **Rule-Based Logic** for deterministic severity assessment
- **Knowledge Grounding** (JSON databases) to prevent hallucination

### Design Principles
1. **Explainability** — Every diagnosis can be traced through the 8-node pipeline
2. **Determinism** — Severity is rule-based, not LLM-generated
3. **Safety-First** — Organic solutions preferred, harmful advice filtered
4. **Graceful Degradation** — Unknown plants get generic care, not errors

---

## Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Runtime | Python 3.12+ | Modern Python with type hints |
| Package Manager | uv | Fast dependency resolution |
| Reasoning Framework | LangGraph 0.2+ | Stateful agent orchestration |
| LLM Provider | Gemini (via langchain-google-genai) | Cause analysis & pro tips |
| Object Detection | Ultralytics YOLOv8 | Plant symptom detection |
| Web Framework | Streamlit 1.40+ | Interactive UI |
| Validation | Pydantic 2.0+ | State model validation |

### Environment Variables
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

---

## LangGraph Pipeline

### Flow Diagram

```
START
  │
  ▼
┌─────────────────┐
│ 1. Identification│──▶ Confirm species or mark "Unknown"
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Symptoms     │──▶ Map YOLO labels to stress categories
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Severity     │──▶ RULE-BASED: Critical/Moderate/Mild/Healthy
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Causes       │──▶ LLM explains biological reasons
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. Seasonal     │──▶ Adjust for current season
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 6. Care Plan    │──▶ Immediate + ongoing actions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 7. Safety       │──▶ Filter harmful advice
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 8. Formatter    │──▶ Assemble final response
└────────┬────────┘
         │
         ▼
        END
```

### Node Responsibilities

#### Node 1: Identification
- Input: `image`, `plant_id_confidence`
- Output: `plant_name`
- Logic: If confidence < 0.6, set `plant_name = "Unknown"`

#### Node 2: Symptoms
- Input: `yolo_detections`
- Output: `symptoms_grouped`
- Logic: Group labels by category (nutrient, water, pest, fungal, light)

#### Node 3: Severity
- Input: `symptoms_grouped`, `yolo_detections`
- Output: `severity`, `is_healthy`
- Logic: **Deterministic calculation** based on weights & categories

#### Node 4: Causes
- Input: `symptoms_grouped`, `plant_name`
- Output: `causes`
- Logic: LLM-powered reasoning (Gemini)

#### Node 5: Seasonal
- Input: `season`, `symptoms_grouped`
- Output: `seasonal_insight`
- Logic: Adjust recommendations for current season

#### Node 6: Care Plan
- Input: `severity`, `causes`, `is_healthy`
- Output: `care_immediate`, `care_ongoing`
- Logic: Generate actionable steps

#### Node 7: Safety
- Input: `care_immediate`, `care_ongoing`
- Output: `dont_do`
- Logic: Add warnings, prefer organic solutions

#### Node 8: Formatter
- Input: All state fields
- Output: `final_response`, `rescan_suggested`
- Logic: Assemble markdown output with emoji headers

---

## State Management

### PlantState Model

```python
class YOLODetection(BaseModel):
    label: str
    confidence: float

class PlantState(BaseModel):
    # Input
    image: Optional[bytes] = None
    season: str = "unknown"
    
    # Detection results
    plant_name: str = "Unknown"
    plant_id_confidence: float = 0.0
    yolo_detections: List[YOLODetection] = []
    
    # Reasoning outputs
    severity: Optional[str] = None
    confidence: Optional[str] = None
    is_healthy: bool = False
    symptoms_grouped: Dict[str, list] = {}
    causes: List[str] = []
    
    # Care recommendations
    care_immediate: List[str] = []
    care_ongoing: List[str] = []
    dont_do: List[str] = []
    seasonal_insight: Optional[str] = None
    pro_tip: Optional[str] = None
    
    # Output
    rescan_suggested: bool = False
    final_response: Optional[str] = None
```

### Confidence Calculation
```python
def calculate_confidence(state: PlantState) -> str:
    factors = [
        state.plant_id_confidence > 0.7,
        len(state.yolo_detections) >= 2,
        all(d.confidence > 0.5 for d in state.yolo_detections)
    ]
    score = sum(factors)
    if score >= 3:
        return "High"
    elif score >= 2:
        return "Medium"
    return "Low"
```

---

## Knowledge Base

### symptoms.json
Maps YOLO labels to plant stress patterns:

```json
{
  "leaf_yellowing": {
    "category": "nutrient",
    "possible_causes": ["nitrogen_deficiency", "overwatering"],
    "severity_weight": 2
  },
  "brown_spots": {
    "category": "fungal",
    "possible_causes": ["leaf_spot_disease", "bacterial_blight"],
    "severity_weight": 3
  },
  "wilting": {
    "category": "water",
    "possible_causes": ["underwatering", "root_rot"],
    "severity_weight": 2
  }
}
```

### plants.json
Plant-specific care information:

```json
{
  "pothos": {
    "scientific_name": "Epipremnum aureum",
    "light": "indirect",
    "water_frequency": "weekly",
    "common_issues": ["root_rot", "yellowing"],
    "pro_tips": ["Tolerates low light", "Prune to encourage bushiness"]
  }
}
```

### seasons.json
Seasonal adjustments:

```json
{
  "winter": {
    "watering_modifier": 0.7,
    "fertilizing": false,
    "warnings": ["Reduced growth is normal", "Keep away from cold drafts"]
  },
  "summer": {
    "watering_modifier": 1.3,
    "fertilizing": true,
    "warnings": ["Increase humidity", "Watch for pests"]
  }
}
```

---

## API Reference

### Running a Diagnosis

```python
from floravision.graph import create_graph
from floravision.state import PlantState

# Initialize graph
graph = create_graph()

# Create initial state
state = PlantState(
    image=image_bytes,
    season="winter"
)

# Run diagnosis
result = graph.invoke(state)
print(result.final_response)
```

### Detection API

```python
from floravision.detection import YOLODetector

detector = YOLODetector()
detections = detector.detect(image_bytes)
# Returns: List[YOLODetection]
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Low confidence diagnoses | Suggest rescan with better lighting |
| Unknown plant | Generic care mode activates automatically |
| No symptoms detected | Healthy plant path with maintenance tips |
| Conflicting symptoms | Rescan suggested, confidence lowered |
