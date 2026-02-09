from .base import BaseLLM
from .gemini import GeminiLLM
from .groq import GroqLLM

__all__ = ["BaseLLM", "GeminiLLM", "GroqLLM", "get_llm"]


# Singleton instances
_groq_instance = None
_gemini_instance = None


def get_llm() -> BaseLLM:
    """
    Get an LLM instance with tiered fallback logic:
    1. Try Groq (Primary)
    2. Try Gemini (Fallback)
    
    Returns:
        The first available LLM provider.
    """
    global _groq_instance, _gemini_instance
    
    # 1. Try Groq
    if _groq_instance is None:
        _groq_instance = GroqLLM()
    
    if _groq_instance.is_available:
        return _groq_instance
    
    # 2. Try Gemini (Fallback)
    if _gemini_instance is None:
        _gemini_instance = GeminiLLM()
    
    if _gemini_instance.is_available:
        # Note: If this hits quota (429), it will return None in its methods
        # which will trigger fallbacks in nodes (e.g., knowledge base lookup)
        return _gemini_instance
    
    # Final fallback: Return Gemini instance even if key is missing 
    # (it will log clean errors when used)
    return _gemini_instance or _groq_instance
