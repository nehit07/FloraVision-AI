"""
FloraVision AI - Chat Tests
===========================

Tests the ChatManager contextual responses.
"""

import pytest
from src.floravision.utils.chat_manager import ChatManager
from src.floravision.state import PlantState, YOLODetection

@pytest.fixture
def chat():
    return ChatManager()

@pytest.fixture
def mock_state():
    return PlantState(
        plant_name="pothos",
        severity="Mild",
        is_healthy=False,
        yolo_detections=[YOLODetection(label="leaf_yellowing", confidence=0.8)],
        care_immediate=["Check soil moisture"],
        climate_zone="Tropical",
        season="summer",
        final_response="## 🩺 Health Assessment\nYour pothos has minor yellowing."
    )

def test_chat_response_generation(chat, mock_state):
    """Test that chat manager returns a response."""
    # This test will only pass if GOOGLE_API_KEY is set, 
    # otherwise it returns the unavailable message.
    response = chat.get_response("Is this safe for my cat?", mock_state)
    assert response is not None
    assert len(response) > 10

def test_chat_unavailable_without_api(chat, mock_state, monkeypatch):
    """Test the fallback message when LLM is unavailable."""
    # Force is_available to False by patching the internal _model
    monkeypatch.setattr(chat.llm, "_model", None)
    response = chat.get_response("Any tips?", mock_state)
    assert "unavailable" in response.lower()
