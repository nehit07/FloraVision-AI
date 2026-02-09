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
        self.model = None
        
        # Load known plants from knowledge base
        with open(PLANTS_PATH) as f:
            self.plants_data = json.load(f)
        self.known_plants = list(self.plants_data.keys())
        
        # Try to initialize Gemini
        if not mock:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                api_key = os.getenv("GOOGLE_API_KEY")
                if api_key:
                    self.model = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        google_api_key=api_key
                    )
                    self.mock = False
                else:
                    print("Warning: GOOGLE_API_KEY not found. Using mock mode.")
                    self.mock = True
            except Exception as e:
                print(f"Warning: Could not initialize Gemini: {e}")
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
        
        Returns a random known plant with varying confidence.
        """
        import random
        
        # 70% chance of recognizing a known plant
        if random.random() < 0.7:
            plant = random.choice([p for p in self.known_plants if p != "unknown"])
            confidence = random.uniform(0.65, 0.95)
            return plant, round(confidence, 2)
        else:
            # Return unknown with low confidence
            return "unknown", round(random.uniform(0.3, 0.55), 2)
    
    def _real_identify(self, image_bytes: bytes) -> Tuple[str, float]:
        """
        Real identification using Gemini Vision.
        """
        try:
            from langchain_core.messages import HumanMessage
            
            # Encode image to base64
            image_b64 = base64.b64encode(image_bytes).decode()
            
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

            # Create message with image
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                    }
                ]
            )
            
            # Get response
            response = self.model.invoke([message])
            
            # Parse response
            return self._parse_response(response.content)
            
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
