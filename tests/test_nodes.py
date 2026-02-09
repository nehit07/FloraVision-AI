"""
FloraVision AI - Unit Tests for Reasoning Nodes
=================================================

Tests each of the 8 reasoning nodes independently.
Uses mock PlantState objects to verify node behavior.

Run with: uv run pytest tests/test_nodes.py -v
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from floravision.state import PlantState, YOLODetection, calculate_confidence
from floravision.nodes.identification import identification_node
from floravision.nodes.symptoms import symptoms_node, get_symptom_display_name
from floravision.nodes.severity import severity_node
from floravision.nodes.seasonal import seasonal_node, get_season_from_month
from floravision.nodes.care_plan import care_plan_node
from floravision.nodes.safety import safety_node
from floravision.nodes.formatter import formatter_node


# ═══════════════════════════════════════════════════════════════════
# TEST FIXTURES
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def healthy_state():
    """A state representing a healthy plant with no symptoms."""
    return PlantState(
        plant_name="pothos",
        plant_id_confidence=0.85,
        yolo_detections=[],
        season="spring"
    )


@pytest.fixture
def mild_state():
    """A state with mild symptoms."""
    return PlantState(
        plant_name="monstera",
        plant_id_confidence=0.78,
        yolo_detections=[
            YOLODetection(label="brown_tips", confidence=0.72)
        ],
        season="summer"
    )


@pytest.fixture
def critical_state():
    """A state with critical symptoms (fungal)."""
    return PlantState(
        plant_name="peace_lily",
        plant_id_confidence=0.92,
        yolo_detections=[
            YOLODetection(label="powdery_mildew", confidence=0.88),
            YOLODetection(label="wilting", confidence=0.75)
        ],
        season="autumn"
    )


@pytest.fixture
def unknown_plant_state():
    """A state with low confidence plant identification."""
    return PlantState(
        plant_name="some_plant",
        plant_id_confidence=0.45,  # Below threshold
        yolo_detections=[
            YOLODetection(label="leaf_yellowing", confidence=0.65)
        ],
        season="winter"
    )


# ═══════════════════════════════════════════════════════════════════
# NODE 1: IDENTIFICATION TESTS
# ═══════════════════════════════════════════════════════════════════

class TestIdentificationNode:
    """Tests for Node 1: Plant Identification."""
    
    def test_valid_plant_kept(self, healthy_state):
        """High confidence, known plant should be kept."""
        result = identification_node(healthy_state)
        assert result["plant_name"] == "pothos"
    
    def test_low_confidence_becomes_unknown(self, unknown_plant_state):
        """Low confidence should become 'unknown'."""
        result = identification_node(unknown_plant_state)
        assert result["plant_name"] == "unknown"
    
    def test_unknown_plant_in_db(self):
        """A plant not in database should become 'unknown'."""
        state = PlantState(
            plant_name="rare_exotic_plant",
            plant_id_confidence=0.9,
            season="spring"
        )
        result = identification_node(state)
        assert result["plant_name"] == "unknown"


# ═══════════════════════════════════════════════════════════════════
# NODE 2: SYMPTOMS TESTS
# ═══════════════════════════════════════════════════════════════════

class TestSymptomsNode:
    """Tests for Node 2: Symptom Interpretation."""
    
    def test_empty_detections(self, healthy_state):
        """No detections should produce empty grouped symptoms."""
        result = symptoms_node(healthy_state)
        assert result["symptoms_grouped"] == {}
    
    def test_single_symptom_grouped(self, mild_state):
        """Single symptom should be grouped correctly."""
        result = symptoms_node(mild_state)
        assert "water" in result["symptoms_grouped"]
        assert "brown_tips" in result["symptoms_grouped"]["water"]
    
    def test_multiple_symptoms_grouped(self, critical_state):
        """Multiple symptoms should be grouped by category."""
        result = symptoms_node(critical_state)
        assert "fungal" in result["symptoms_grouped"]
        assert "water" in result["symptoms_grouped"]
    
    def test_symptom_display_name(self):
        """Display name should be human readable."""
        display = get_symptom_display_name("leaf_yellowing")
        assert display == "Leaf Yellowing"


# ═══════════════════════════════════════════════════════════════════
# NODE 3: SEVERITY TESTS
# ═══════════════════════════════════════════════════════════════════

class TestSeverityNode:
    """Tests for Node 3: Severity Assessment."""
    
    def test_healthy_plant(self, healthy_state):
        """No symptoms should result in healthy plant."""
        result = severity_node(healthy_state)
        assert result["severity"] is None
        assert result["is_healthy"] is True
    
    def test_mild_severity(self, mild_state):
        """Single low-weight symptom should be Mild."""
        result = severity_node(mild_state)
        assert result["severity"] == "Mild"
        assert result["is_healthy"] is False
    
    def test_critical_severity_fungal(self, critical_state):
        """Fungal symptoms should trigger Critical severity."""
        result = severity_node(critical_state)
        assert result["severity"] == "Critical"
    
    def test_confidence_calculation(self, critical_state):
        """Confidence should be calculated."""
        result = severity_node(critical_state)
        assert result["confidence"] in ["High", "Medium", "Low"]


# ═══════════════════════════════════════════════════════════════════
# NODE 5: SEASONAL TESTS
# ═══════════════════════════════════════════════════════════════════

class TestSeasonalNode:
    """Tests for Node 5: Seasonal Context."""
    
    def test_winter_insight(self):
        """Winter should produce seasonal insight."""
        state = PlantState(
            plant_name="pothos",
            season="winter",
            symptoms_grouped={}
        )
        result = seasonal_node(state)
        assert result["seasonal_insight"] is not None
        assert "winter" in result["seasonal_insight"].lower() or "dormancy" in result["seasonal_insight"].lower()
    
    def test_season_from_month(self):
        """Months should map to correct seasons."""
        assert get_season_from_month(1) == "winter"
        assert get_season_from_month(4) == "spring"
        assert get_season_from_month(7) == "summer"
        assert get_season_from_month(10) == "autumn"


# ═══════════════════════════════════════════════════════════════════
# NODE 6: CARE PLAN TESTS
# ═══════════════════════════════════════════════════════════════════

class TestCarePlanNode:
    """Tests for Node 6: Care Plan Generation."""
    
    def test_healthy_plant_care(self, healthy_state):
        """Healthy plant should get maintenance care."""
        healthy_state.is_healthy = True
        result = care_plan_node(healthy_state)
        assert len(result["care_immediate"]) > 0
        assert len(result["care_ongoing"]) > 0
    
    def test_symptomatic_plant_care(self, critical_state):
        """Symptomatic plant should get targeted care."""
        critical_state.symptoms_grouped = {"fungal": ["powdery_mildew"], "water": ["wilting"]}
        result = care_plan_node(critical_state)
        assert len(result["care_immediate"]) > 0
        # Should mention isolation or removal for fungal
        immediate_text = " ".join(result["care_immediate"]).lower()
        assert "isolate" in immediate_text or "remove" in immediate_text


# ═══════════════════════════════════════════════════════════════════
# NODE 7: SAFETY TESTS
# ═══════════════════════════════════════════════════════════════════

class TestSafetyNode:
    """Tests for Node 7: Safety Filter."""
    
    def test_produces_warnings(self, mild_state):
        """Should produce don't do warnings."""
        mild_state.symptoms_grouped = {"water": ["brown_tips"]}
        result = safety_node(mild_state)
        assert len(result["dont_do"]) > 0
    
    def test_produces_pro_tip(self, healthy_state):
        """Should produce a pro tip."""
        result = safety_node(healthy_state)
        assert result["pro_tip"] is not None


# ═══════════════════════════════════════════════════════════════════
# NODE 8: FORMATTER TESTS
# ═══════════════════════════════════════════════════════════════════

class TestFormatterNode:
    """Tests for Node 8: Response Formatting."""
    
    def test_produces_markdown(self, healthy_state):
        """Should produce markdown response."""
        # Populate required fields
        healthy_state.is_healthy = True
        healthy_state.confidence = "High"
        healthy_state.care_immediate = ["Keep doing what you're doing"]
        healthy_state.care_ongoing = ["Water weekly"]
        healthy_state.dont_do = ["Don't overwater"]
        healthy_state.seasonal_insight = "Spring is growth time"
        healthy_state.pro_tip = "Propagate in water"
        
        result = formatter_node(healthy_state)
        
        # Check all required sections exist
        response = result["final_response"]
        assert "## 🩺 Health Assessment" in response
        assert "## 📋 Treatment Plan" in response
        assert "## ⚠️ Common Mistakes to Avoid" in response
        assert "Seasonal Care" in response
        assert "## 💡 Expert Tip" in response
    
    def test_rescan_suggested_low_confidence(self):
        """Low confidence should suggest rescan."""
        state = PlantState(
            plant_name="unknown",
            confidence="Low",
            yolo_detections=[YOLODetection(label="leaf_yellowing", confidence=0.4)],
            care_immediate=["Check soil"],
            care_ongoing=["Water weekly"],
            dont_do=["Don't panic"],
            seasonal_insight="Check conditions",
            pro_tip="Observe carefully"
        )
        result = formatter_node(state)
        assert result["rescan_suggested"] is True


# ═══════════════════════════════════════════════════════════════════
# STATE MODEL TESTS
# ═══════════════════════════════════════════════════════════════════

class TestStateModels:
    """Tests for state models."""
    
    def test_yolo_detection_validation(self):
        """YOLODetection should validate confidence range."""
        detection = YOLODetection(label="test", confidence=0.5)
        assert detection.confidence == 0.5
    
    def test_plant_state_defaults(self):
        """PlantState should have correct defaults."""
        state = PlantState()
        assert state.plant_name == "Unknown"
        assert state.yolo_detections == []
        assert state.is_healthy is False
    
    def test_confidence_calculation_high(self):
        """High confidence calculation."""
        state = PlantState(
            plant_id_confidence=0.8,
            yolo_detections=[
                YOLODetection(label="a", confidence=0.7),
                YOLODetection(label="b", confidence=0.8)
            ]
        )
        assert calculate_confidence(state) == "High"
    
    def test_confidence_calculation_low(self):
        """Low confidence calculation."""
        state = PlantState(
            plant_id_confidence=0.3,
            yolo_detections=[]
        )
        assert calculate_confidence(state) == "Low"
