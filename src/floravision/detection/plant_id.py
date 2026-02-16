"""
FloraVision AI - Plant Species Identification
==============================================

PURPOSE:
    Identifies plant species from images using Gemini Vision API.
    Falls back to "Unknown" if confidence is low.

CONNECTIONS:
    ┌────────────────────────────────────────────────────────────────┐
    │                    Plant Identifier                            │
    │                         │                                      │
    │   Reads:                │    Writes to:                        │
    │   - Image from state    │    - state.plant_name                │
    │                         │    - state.plant_id_confidence       │
    │                         │                                      │
    │   Used by:              │    Uses:                             │
    │   - graph.py (before    │    - langchain-google-genai          │
    │     pipeline starts)    │    - knowledge/plants.json           │
    └────────────────────────────────────────────────────────────────┘

FALLBACK RULES:
    - If confidence < 0.6 → plant_name = "Unknown"
    - Unknown plants get generic care recommendations
    - Always explain uncertainty in final response
"""

import json
import base64
import os
import hashlib
import random
from pathlib import Path
from typing import Tuple, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Path to plants knowledge base
PLANTS_PATH = Path(__file__).parent.parent / "knowledge" / "plants.json"


class PlantIdentifier:
    """
    Identifies plant species using Gemini Vision API.
    
    Falls back gracefully to "Unknown" if:
    - API is unavailable
    - Confidence is too low
    - Plant is not in our knowledge base
    
    Usage:
        identifier = PlantIdentifier()
        name, confidence = identifier.identify(image_bytes)
    """
    
    def __init__(self, mock: bool = True):
        """
        Initialize the plant identifier.
        
        Args:
            mock: If True, use mock identification for demo
        """
        self.mock = mock
        self.llm = None
        
        # Load known plants from knowledge base
        with open(PLANTS_PATH) as f:
            self.plants_data = json.load(f)
        self.known_plants = list(self.plants_data.keys())
        
        # Try to initialize LLM abstraction
        if not mock:
            from ..llm import get_llm
            self.llm = get_llm()
            if not self.llm.is_available:
                print("Warning: LLM not available. Using mock mode.")
                self.mock = True
    
    def identify(self, image_bytes: bytes) -> Tuple[str, float]:
        """
        Identify the plant species in the image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Tuple of (plant_name, confidence)
            - plant_name: Identified species or "Unknown"
            - confidence: 0.0 to 1.0
        """
        if self.mock:
            return self._mock_identify(image_bytes)
        else:
            return self._real_identify(image_bytes)
    
    def _mock_identify(self, image_bytes: bytes) -> Tuple[str, float]:
        """
        Mock identification for demo/development.
        
        Returns a deterministic known plant based on image hash.
        """
        # Create a deterministic seed from image bytes
        seed = int(hashlib.sha256(image_bytes).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        
        # 70% chance of recognizing a known plant
        if rng.random() < 0.7:
            plant = rng.choice([p for p in self.known_plants if p != "unknown"])
            confidence = rng.uniform(0.65, 0.95)
            return plant, round(confidence, 2)
        else:
            # Return unknown with low confidence
            return "unknown", round(rng.uniform(0.3, 0.55), 2)
    
    def _real_identify(self, image_bytes: bytes) -> Tuple[str, float]:
        """
        Real identification using LLM abstraction.
        """
        try:
            # Create prompt for plant identification
            prompt = f"""Identify this plant. 

Known plants in our database: {', '.join(self.known_plants[:-1])}

Respond in this exact format:
PLANT: <plant_name>
CONFIDENCE: <0.0 to 1.0>

If you cannot identify the plant or it's not in the list, respond with:
PLANT: unknown
CONFIDENCE: 0.3

Only respond with the two lines above, nothing else."""

            # Use LLM with image support
            response_text = self.llm.generate_with_image(prompt, image_bytes)
            
            if not response_text:
                return "unknown", 0.3
            
            # Parse response
            return self._parse_response(response_text)
            
        except Exception as e:
            print(f"Plant identification error: {e}")
            return "unknown", 0.3
    
    def _parse_response(self, response: str) -> Tuple[str, float]:
        """
        Parse the Gemini response into plant name and confidence.
        """
        try:
            lines = response.strip().split('\n')
            plant_line = lines[0].replace("PLANT:", "").strip().lower()
            conf_line = lines[1].replace("CONFIDENCE:", "").strip()
            
            confidence = float(conf_line)
            
            # Validate plant is known
            if plant_line in self.known_plants:
                return plant_line, confidence
            else:
                return "unknown", min(confidence, 0.5)
                
        except Exception:
            return "unknown", 0.3


def identify_plant(image_bytes: bytes, mock: bool = True) -> Tuple[str, float]:
    """
    Quick identification function.
    
    Args:
        image_bytes: Raw image bytes
        mock: Use mock mode for demo
        
    Returns:
        Tuple of (plant_name, confidence)
    """
    identifier = PlantIdentifier(mock=mock)
    return identifier.identify(image_bytes)
