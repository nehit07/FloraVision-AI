# 🌿 FloraVision AI

**An intelligent plant health assistant powered by computer vision and agentic reasoning.**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red.svg)](https://streamlit.io)

---

## 🎯 What is FloraVision AI?

FloraVision AI analyzes plant images through a multi-step reasoning pipeline to diagnose health issues and provide actionable care recommendations. Think of it as having a plant pathologist, professional gardener, and AI reasoning agent working together.

### Key Features

- 🔍 **YOLO-powered symptom detection** — Identifies yellowing, spots, wilting, pests, and more
- 🧠 **8-node LangGraph reasoning pipeline** — Structured diagnosis flow from identification to care plan
- 📸 **Camera & upload support** — Scan plants live or upload existing photos
- 🌤️ **Seasonal awareness** — Adjusts recommendations based on current season
- ✅ **Rule-based severity** — Deterministic assessment before LLM reasoning
- 🛡️ **Safety-first approach** — Prefers organic, low-risk solutions

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────────────────────┐
│  📷 Image   │────▶│  YOLO v8     │────▶│  LangGraph Pipeline        │
│   Input     │     │  Detection   │     │  ┌────────────────────────┐│
└─────────────┘     └──────────────┘     │  │ 1. Plant ID            ││
                                         │  │ 2. Symptom Mapping     ││
                                         │  │ 3. Severity Assessment ││
                                         │  │ 4. Cause Analysis      ││
                                         │  │ 5. Seasonal Context    ││
                                         │  │ 6. Care Plan           ││
                                         │  │ 7. Safety Filter       ││
                                         │  │ 8. Response Format     ││
                                         │  └────────────────────────┘│
                                         └────────────────────────────┘
                                                      │
                                                      ▼
                                         ┌────────────────────────────┐
                                         │  📄 Diagnosis Report       │
                                         │  • Severity assessment     │
                                         │  • Care recommendations    │
                                         │  • Seasonal insights       │
                                         └────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/floravision-ai.git
cd floravision-ai

# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Add your GOOGLE_API_KEY for Gemini
```

### Running the App

```bash
uv run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
floravision-ai/
├── src/floravision/
│   ├── state.py              # Pydantic state models
│   ├── graph.py              # LangGraph pipeline
│   ├── nodes/                # 8 reasoning nodes
│   │   ├── identification.py
│   │   ├── symptoms.py
│   │   ├── severity.py
│   │   ├── causes.py
│   │   ├── seasonal.py
│   │   ├── care_plan.py
│   │   ├── safety.py
│   │   └── formatter.py
│   ├── detection/            # CV models
│   │   ├── yolo_detector.py
│   │   └── plant_id.py
│   └── knowledge/            # Grounded reasoning data
│       ├── symptoms.json
│       ├── plants.json
│       └── seasons.json
├── app.py                    # Streamlit entry point
├── tests/                    # Test suites
└── pyproject.toml
```

---

## 🔬 How It Works

### 1. Image Input
User captures or uploads a plant photo.

### 2. YOLO Detection
YOLOv8 identifies visual symptoms with confidence scores:
```json
[
  {"label": "leaf_yellowing", "confidence": 0.82},
  {"label": "brown_spots", "confidence": 0.67}
]
```

### 3. LangGraph Reasoning
The 8-node pipeline processes the detections:

| Node | Purpose |
|------|---------|
| Plant ID | Identify species (fallback to "Unknown" if confidence < 60%) |
| Symptoms | Group labels by category (nutrient, fungal, pest, etc.) |
| Severity | **Rule-based** classification: Critical/Moderate/Mild/Healthy |
| Causes | LLM-powered causal reasoning |
| Seasonal | Adjust for current season (e.g., winter dormancy) |
| Care Plan | Generate immediate + ongoing actions |
| Safety | Filter harmful advice, prefer organic |
| Format | Assemble user-friendly report |

### 4. Output
Structured diagnosis with care recommendations:
- 🩺 Plant Diagnosis
- 📝 Care Plan (Do This Now + Ongoing)
- 🚫 What Not To Do
- 🌤️ Seasonal Insight
- 💡 Pro Tip

---

## 🧪 Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Specific modules
uv run pytest tests/test_nodes.py -v    # Node logic
uv run pytest tests/test_graph.py -v    # Graph integration
```

---

## 🌱 Example Output

```markdown
## 🩺 Plant Diagnosis
• Identified Plant: Pothos (Epipremnum aureum)
• Detected Symptoms: Leaf yellowing (nutrient stress), brown tips (underwatering)
• Severity: Moderate
• Confidence Level: High

---

## 📝 Care Plan
**Do This Now**
- Move plant away from direct sunlight
- Water thoroughly until drainage

**Ongoing Care**
- Check soil moisture weekly
- Apply balanced liquid fertilizer monthly

---

## 🚫 What Not To Do
- Don't overwater — root rot is worse than underwatering
- Avoid repotting while the plant is stressed

---

## 🌤️ Seasonal Insight
Winter months naturally slow growth. Reduce watering frequency.

---

## 💡 Pro Tip
Pothos love humidity! Mist leaves or place near a humidifier.
```

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

## 🤝 Contributing

Contributions welcome! Please read the [Contributing Guide](CONTRIBUTING.md) first.

---

<p align="center">
  Made with 🌿 for healthier plants
</p>
