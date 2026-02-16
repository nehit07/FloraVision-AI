"""
FloraVision AI - Database Manager
==================================

PURPOSE:
    Handles persistent storage for plant diagnoses using SQLite.
    Stores metadata in DB and large images in the filesystem.
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from ..state import PlantState, YOLODetection

# Define project-relative paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
DB_PATH = BASE_DIR / "data" / "floravision.db"
HISTORY_IMAGES_DIR = BASE_DIR / "data" / "history_images"

class DatabaseManager:
    """
    Manages SQLite database for storing diagnosis history.
    """
    
    def __init__(self):
        """Initialize database and ensure tables exist."""
        # Ensure directories exist
        os.makedirs(BASE_DIR / "data", exist_ok=True)
        os.makedirs(HISTORY_IMAGES_DIR, exist_ok=True)
        
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        
    def _create_tables(self):
        """Create the diagnoses table if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS diagnoses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                plant_name TEXT NOT NULL,
                severity TEXT,
                confidence TEXT,
                is_healthy INTEGER,
                symptoms_json TEXT,
                yolo_detections_json TEXT,
                care_immediate_json TEXT,
                care_ongoing_json TEXT,
                final_response TEXT,
                image_path TEXT
            )
        ''')
        self.conn.commit()
        
    def save_diagnosis(self, state: PlantState) -> int:
        """
        Save a PlantState to the database.
        
        Args:
            state: The PlantState object to persist
            
        Returns:
            The ID of the saved record
        """
        timestamp = datetime.now().isoformat()
        
        # Save image to file system if present
        image_path = None
        if state.image:
            filename = f"diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            image_path = os.path.relpath(HISTORY_IMAGES_DIR / filename, BASE_DIR)
            with open(BASE_DIR / image_path, "wb") as f:
                f.write(state.image)
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO diagnoses (
                timestamp, plant_name, severity, confidence, 
                is_healthy, symptoms_json, yolo_detections_json, 
                care_immediate_json, care_ongoing_json, final_response, 
                image_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp,
            state.plant_name,
            state.severity,
            state.diagnosis_confidence,
            1 if state.is_healthy else 0,
            json.dumps(state.symptoms_grouped),
            json.dumps([d.model_dump() for d in state.yolo_detections]),
            json.dumps(state.care_immediate),
            json.dumps(state.care_ongoing),
            state.final_response,
            image_path
        ))
        
        new_id = cursor.lastrowid
        self.conn.commit()
        return new_id
        
    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent diagnosis history."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM diagnoses 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
        
    def get_diagnosis_by_id(self, diagnosis_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific diagnosis by its ID."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM diagnoses WHERE id = ?', (diagnosis_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def delete_diagnosis(self, diagnosis_id: int):
        """Delete a diagnosis and its associated image."""
        diagnosis = self.get_diagnosis_by_id(diagnosis_id)
        if diagnosis and diagnosis['image_path']:
            image_full_path = BASE_DIR / diagnosis['image_path']
            if image_full_path.exists():
                os.remove(image_full_path)
                
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM diagnoses WHERE id = ?', (diagnosis_id,))
        self.conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the dashboard."""
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM diagnoses')
        total = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM diagnoses WHERE is_healthy = 1')
        healthy = cursor.fetchone()[0]
        
        cursor.execute('SELECT severity, COUNT(*) as count FROM diagnoses GROUP BY severity')
        severity_dist = {row[0]: row[1] for row in cursor.fetchall() if row[0]}
        
        return {
            "total_diagnoses": total,
            "healthy_plants": healthy,
            "severity_distribution": severity_dist
        }

# Singleton instance
db_manager = DatabaseManager()
