"""
FloraVision AI - Integration Tests
===================================

Tests the complete LangGraph pipeline end-to-end.
Verifies that all nodes work together correctly.

Run with: uv run pytest tests/test_integration.py -v
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from floravision.state import PlantState, YOLODetection
from floravision.graph import create_graph, run_diagnosis_full


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION TEST FIXTURES
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_image_bytes():
    """Mock image bytes for testing."""
    # Create a minimal valid bytes object (pipeline uses mock mode anyway)
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


@pytest.fixture
def compiled_graph():
    """Get compiled LangGraph for testing."""
    return create_graph().compile()


# ═══════════════════════════════════════════════════════════════════
# PIPELINE INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════

class TestPipelineIntegration:
    """Integration tests for the full LangGraph pipeline."""
    
    def test_healthy_plant_pipeline(self, compiled_graph):
        """Test complete pipeline with healthy plant state."""
        initial_state = PlantState(
            plant_name="pothos",
            plant_id_confidence=0.85,
            yolo_detections=[],
            season="spring"
        )
        
        result = compiled_graph.invoke(initial_state)
        
        # Verify all outputs are populated
        assert result["final_response"] is not None
        assert result["diagnosis_confidence"] is not None
        assert result["is_healthy"] is True
        assert len(result["care_immediate"]) > 0
        assert len(result["care_ongoing"]) > 0
    
    def test_symptomatic_plant_pipeline(self, compiled_graph):
        """Test complete pipeline with symptoms."""
        initial_state = PlantState(
            plant_name="monstera",
            plant_id_confidence=0.88,
            yolo_detections=[
                YOLODetection(label="leaf_yellowing", confidence=0.75),
                YOLODetection(label="brown_tips", confidence=0.68)
            ],
            season="summer"
        )
        
        result = compiled_graph.invoke(initial_state)
        
        # Verify outputs
        assert result["final_response"] is not None
        assert result["severity"] is not None
        assert result["is_healthy"] is False
        assert len(result["symptoms_grouped"]) > 0
        assert len(result["care_immediate"]) > 0
        assert len(result["dont_do"]) > 0
    
    def test_critical_plant_pipeline(self, compiled_graph):
        """Test complete pipeline with critical fungal issues."""
        initial_state = PlantState(
            plant_name="peace_lily",
            plant_id_confidence=0.92,
            yolo_detections=[
                YOLODetection(label="powdery_mildew", confidence=0.85),
                YOLODetection(label="root_rot", confidence=0.78)
            ],
            season="autumn"
        )
        
        result = compiled_graph.invoke(initial_state)
        
        # Verify critical severity handling
        assert result["severity"] == "Critical"
        assert "fungal" in result["symptoms_grouped"]
        # Should have urgent care recommendations
        assert len(result["care_immediate"]) >= 2
    
    def test_unknown_plant_pipeline(self, compiled_graph):
        """Test pipeline with unknown/low confidence plant."""
        initial_state = PlantState(
            plant_name="exotic_rare_plant",
            plant_id_confidence=0.35,  # Low confidence
            yolo_detections=[
                YOLODetection(label="wilting", confidence=0.7)
            ],
            season="winter"
        )
        
        result = compiled_graph.invoke(initial_state)
        
        # Should fall back to unknown plant
        assert result["plant_name"] == "unknown"
        # Should still provide care recommendations
        assert len(result["care_immediate"]) > 0
    
    def test_all_seasons(self, compiled_graph):
        """Test pipeline handles all seasons correctly."""
        for season in ["spring", "summer", "autumn", "winter"]:
            initial_state = PlantState(
                plant_name="pothos",
                plant_id_confidence=0.8,
                yolo_detections=[],
                season=season
            )
            
            result = compiled_graph.invoke(initial_state)
            
            # Should have seasonal insight
            assert result["seasonal_insight"] is not None


# ═══════════════════════════════════════════════════════════════════
# FULL DIAGNOSIS FUNCTION TESTS
# ═══════════════════════════════════════════════════════════════════

class TestRunDiagnosisFull:
    """Tests for the run_diagnosis_full convenience function."""
    
    def test_mock_diagnosis_returns_state(self, sample_image_bytes):
        """Test run_diagnosis_full returns complete state."""
        result = run_diagnosis_full(
            image_bytes=sample_image_bytes,
            season="spring",
            mock=True
        )
        
        # Should be a dict with all state fields
        assert "final_response" in result
        assert "plant_name" in result
        assert "severity" in result
        assert "care_immediate" in result
    
    def test_mock_diagnosis_different_seasons(self, sample_image_bytes):
        """Test diagnosis works with different seasons."""
        for season in ["spring", "summer", "autumn", "winter"]:
            result = run_diagnosis_full(
                image_bytes=sample_image_bytes,
                season=season,
                mock=True
            )
            assert result["final_response"] is not None


# ═══════════════════════════════════════════════════════════════════
# OUTPUT FORMAT TESTS
# ═══════════════════════════════════════════════════════════════════

class TestOutputFormat:
    """Tests for the final output format."""
    
    def test_markdown_sections_present(self, compiled_graph):
        """Verify all required markdown sections are in output."""
        initial_state = PlantState(
            plant_name="pothos",
            plant_id_confidence=0.85,
            yolo_detections=[
                YOLODetection(label="leaf_yellowing", confidence=0.7)
            ],
            season="spring"
        )
        
        result = compiled_graph.invoke(initial_state)
        response = result["final_response"]
        
        # Check mandatory sections
        required_sections = [
            "## 🩺 Health Assessment",
            "## 📋 Treatment Plan",
            "## ⚠️ Common Mistakes to Avoid",
            "Seasonal Care",
            "## 💡 Expert Tip"
        ]
        
        for section in required_sections:
            assert section in response, f"Missing section: {section}"
    
    def test_diagnosis_contains_plant_name(self, compiled_graph):
        """Verify plant name appears in diagnosis."""
        initial_state = PlantState(
            plant_name="monstera",
            plant_id_confidence=0.9,
            yolo_detections=[],
            season="summer"
        )
        
        result = compiled_graph.invoke(initial_state)
        response = result["final_response"]
        
        assert "Monstera" in response or "monstera" in response.lower()
    
    def test_severity_in_output(self, compiled_graph):
        """Verify severity appears in output."""
        initial_state = PlantState(
            plant_name="pothos",
            plant_id_confidence=0.8,
            yolo_detections=[
                YOLODetection(label="powdery_mildew", confidence=0.85)
            ],
            season="spring"
        )
        
        result = compiled_graph.invoke(initial_state)
        response = result["final_response"]
        
        assert "Critical" in response or "Severity" in response
