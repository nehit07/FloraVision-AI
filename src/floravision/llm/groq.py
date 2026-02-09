"""
FloraVision AI - Groq LLM Implementation
==========================================

PURPOSE:
    Implements the BaseLLM interface using Groq's API.
    Used for fast inference and as a primary provider to avoid Gemini quotas.

USAGE:
    from floravision.llm.groq import GroqLLM
    llm = GroqLLM()
    if llm.is_available:
        response = llm.generate("Hello!")
"""

import os
import base64
from typing import Optional
from dotenv import load_dotenv

from .base import BaseLLM

# Load environment variables
load_dotenv()


class GroqLLM(BaseLLM):
    """
    Groq LLM implementation using llama-3.3-70b-versatile or llama-3.2-90b-vision-preview.
    
    Requires GROQ_API_KEY environment variable.
    """
    
    def __init__(self, 
                 model_name: str = "llama-3.3-70b-versatile",
                 vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        """
        Initialize Groq LLM.
        """
        self.model_name = model_name
        self.vision_model = vision_model
        self._model = None
        self._api_key = os.getenv("GROQ_API_KEY")
        
        # Try to initialize the model
        if self._api_key:
            try:
                from langchain_groq import ChatGroq
                self._model = ChatGroq(
                    model=model_name,
                    groq_api_key=self._api_key
                )
            except Exception as e:
                print(f"Warning: Could not initialize Groq: {e}")
                self._model = None
    
    @property
    def is_available(self) -> bool:
        """Check if Groq is properly configured."""
        return self._model is not None
    
    def generate(self, prompt: str) -> Optional[str]:
        """
        Generate text using Groq.
        """
        if not self.is_available:
            return None
        
        try:
            from langchain_core.messages import HumanMessage
            response = self._model.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            print(f"Groq generation error: {e}")
            return None
    
    def generate_with_image(self, prompt: str, image_bytes: bytes) -> Optional[str]:
        """
        Generate text with image using Groq Vision (llama-3.2-90b-vision-preview).
        """
        if not self.is_available:
            return None
        
        try:
            from langchain_groq import ChatGroq
            from langchain_core.messages import HumanMessage
            
            # Use vision model specifically for this request
            vision_model = ChatGroq(
                model=self.vision_model,
                groq_api_key=self._api_key
            )
            
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
            
            response = vision_model.invoke([message])
            return response.content
        except Exception as e:
            print(f"Groq vision error: {e}")
            return None
