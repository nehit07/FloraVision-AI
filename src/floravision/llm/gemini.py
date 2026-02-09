"""
FloraVision AI - Gemini LLM Implementation
============================================

PURPOSE:
    Implements the BaseLLM interface using Google's Gemini model.
    Handles API key configuration and graceful fallback.

USAGE:
    from floravision.llm import get_llm, GeminiLLM
    
    # Using factory function (recommended)
    llm = get_llm()
    if llm.is_available:
        response = llm.generate("Your prompt here")
    
    # Direct instantiation
    llm = GeminiLLM()
"""

import os
import base64
from typing import Optional
from dotenv import load_dotenv

from .base import BaseLLM

# Load environment variables
load_dotenv()


class GeminiLLM(BaseLLM):
    """
    Google Gemini LLM implementation.
    
    Requires GOOGLE_API_KEY environment variable.
    Uses gemini-2.5-flash model for fast, cost-effective responses.
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize Gemini LLM.
        
        Args:
            model_name: Gemini model to use (default: gemini-2.5-flash)
        """
        self.model_name = model_name
        self._model = None
        self._api_key = os.getenv("GOOGLE_API_KEY")
        
        # Try to initialize the model
        if self._api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._model = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=self._api_key
                )
            except Exception as e:
                print(f"Warning: Could not initialize Gemini: {e}")
                self._model = None
    
    @property
    def is_available(self) -> bool:
        """Check if Gemini is properly configured."""
        return self._model is not None
    
    def generate(self, prompt: str) -> Optional[str]:
        """
        Generate text using Gemini.
        
        Args:
            prompt: The text prompt
            
        Returns:
            Generated text, or None if unavailable
        """
        if not self.is_available:
            return None
        
        try:
            from langchain_core.messages import HumanMessage
            response = self._model.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            print(f"Gemini generation error: {e}")
            return None
    
    def generate_with_image(self, prompt: str, image_bytes: bytes) -> Optional[str]:
        """
        Generate text with image using Gemini Vision.
        
        Args:
            prompt: The text prompt
            image_bytes: Raw image bytes
            
        Returns:
            Generated text, or None if unavailable
        """
        if not self.is_available:
            return None
        
        try:
            from langchain_core.messages import HumanMessage
            
            # Encode image to base64
            image_b64 = base64.b64encode(image_bytes).decode()
            
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
            
            response = self._model.invoke([message])
            return response.content
        except Exception as e:
            print(f"Gemini vision error: {e}")
            return None
