"""
FloraVision AI - LLM Base Interface
====================================

PURPOSE:
    Defines the abstract interface for LLM providers.
    All LLM implementations must inherit from BaseLLM.

USAGE:
    from floravision.llm import BaseLLM

    class MyCustomLLM(BaseLLM):
        def generate(self, prompt: str) -> str:
            # Custom implementation
            pass
        
        def generate_with_image(self, prompt: str, image_bytes: bytes) -> str:
            # Custom implementation  
            pass
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.
    
    All LLM implementations must implement:
    - generate(): Text-only generation
    - generate_with_image(): Vision-enabled generation
    
    Properties:
    - is_available: Whether the LLM is properly configured
    """
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the LLM is available and properly configured.
        
        Returns:
            True if the LLM can be used, False otherwise
        """
        pass
    
    @abstractmethod
    def generate(self, prompt: str) -> Optional[str]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The text prompt
            
        Returns:
            Generated text, or None if generation fails
        """
        pass
    
    @abstractmethod
    def generate_with_image(self, prompt: str, image_bytes: bytes) -> Optional[str]:
        """
        Generate text from a prompt with an image.
        
        Args:
            prompt: The text prompt
            image_bytes: Raw image bytes
            
        Returns:
            Generated text, or None if generation fails
        """
        pass
