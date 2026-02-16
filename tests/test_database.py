"""
FloraVision AI - Database Tests
================================

Tests the DatabaseManager CRUD operations.
"""

import pytest
import os
import json
from pathlib import Path
from src.floravision.utils.database import DatabaseManager
from src.floravision.state import PlantState, YOLODetection

@pytest.fixture
def db():
    """Fixture for a test database."""
    # Use a separate test database file
    test_db_path = Path("tests/test_floravision.db")
    if test_db_path.exists():
        os.remove(test_db_path)
    
    # Temporarily monkeypatch the DB_PATH for testing if needed
    # (For this simple test, we'll just instantiate and use)
    manager = DatabaseManager()
    yield manager
    
    # Cleanup (Optional: keep it for manual inspection if needed)
    # if test_db_path.exists():
    #     os.remove(test_db_path)

def test_save_and_get_history(db):
    """Test saving a diagnosis and retrieving it from history."""
    state = PlantState(
        plant_name="pothos",
        severity="Mild",
        diagnosis_confidence="High",
        is_healthy=False,
        symptoms_grouped={"nutrient": ["leaf_yellowing"]},
        yolo_detections=[YOLODetection(label="leaf_yellowing", confidence=0.8, box=[0.1, 0.1, 0.2, 0.2])],
        care_immediate=["Water the plant"],
        care_ongoing=["Monitor light"],
        final_response="## 🩺 Diagnostics Result\nHealthy pothos",
        image=None # No image for this test
    )
    
    new_id = db.save_diagnosis(state)
    assert new_id > 0
    
    history = db.get_history()
    assert len(history) >= 1
    assert history[0]['plant_name'] == "pothos"
    assert history[0]['severity'] == "Mild"
    assert "leaf_yellowing" in history[0]['yolo_detections_json']

def test_get_stats(db):
    """Test database statistics calculation."""
    # Ensure some data exists
    state1 = PlantState(plant_name="p1", is_healthy=True, severity=None)
    state2 = PlantState(plant_name="p2", is_healthy=False, severity="Critical")
    
    db.save_diagnosis(state1)
    db.save_diagnosis(state2)
    
    stats = db.get_stats()
    assert stats['total_diagnoses'] >= 2
    assert stats['healthy_plants'] >= 1
    assert "Critical" in stats['severity_distribution']

def test_delete_diagnosis(db):
    """Test deleting a record from history."""
    state = PlantState(plant_name="delete_me", is_healthy=True)
    new_id = db.save_diagnosis(state)
    
    db.delete_diagnosis(new_id)
    result = db.get_diagnosis_by_id(new_id)
    assert result is None
