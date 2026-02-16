"""
FloraVision AI - Chat Manager
=============================

PURPOSE:
    Handles follow-up Q&A for plant diagnoses.
    Provides contextual answers based on the initial diagnosis.
"""

from typing import List, Dict, Optional
from ..llm.gemini import GeminiLLM
from ..state import PlantState

class ChatManager:
    """
    Manages conversational interaction after a diagnosis.
    """
    
    def __init__(self):
        self.llm = GeminiLLM()
        
    def get_response(self, question: str, state: PlantState, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Get a contextual response to a follow-up question.
        
        Args:
            question: The user's question
            state: The PlantState from the original diagnosis
            chat_history: List of previous messages in the session
            
        Returns:
            Botanist-style contextual answer
        """
        if not self.llm.is_available:
            return "I'm sorry, I cannot answer questions right now as the AI assistant is unavailable."
            
        # Build context from state
        context = f"""
        CONTEXT:
        Plant: {state.plant_name}
        Severity: {state.severity}
        Detections: {', '.join([d.label for d in state.yolo_detections])}
        Health Check: {'Healthy' if state.is_healthy else 'Problematic'}
        Immediate Care: {', '.join(state.care_immediate)}
        Climate Zone: {state.climate_zone}
        Season: {state.season}
        
        Original Diagnosis Summary:
        {state.final_response[:500]}...
        """
        
        # Build history string
        history_str = ""
        if chat_history:
            for msg in chat_history:
                role = "User" if msg['role'] == "user" else "Botanist"
                history_str += f"{role}: {msg['content']}\n"
        
        prompt = f"""
        You are the FloraVision AI Professional Botanist. A user has just received a diagnosis for their plant and has follow-up questions.
        
        {context}
        
        {history_str}
        User Question: {question}
        
        INSTRUCTIONS:
        1. Answer the question specifically using the context above.
        2. Keep the tone helpful, professional, and encouraging.
        3. If the user asks about something not in the context (like "what is your favorite movie?"), politely bring the conversation back to their plant.
        4. If the user asks for more specific care steps, provide actionable botanical advice.
        5. Limit the response to 3-4 concise paragraphs.
        
        Botanist Response:
        """
        
        response = self.llm.generate(prompt)
        return response or "I'm having trouble thinking of an answer right now. Please try again in a moment."

# Singleton instance
chat_manager = ChatManager()
