"""
Meeting Agent AI Module

This module provides an AI-powered meeting assistant that:
- Tracks meeting agenda and generates TODO lists
- Analyzes transcripts in real-time
- Suggests follow-up questions
- Generates meeting summaries
"""

from .meeting_agent import MeetingAgent, AgentState, TodoItem, TodoStatus, Suggestion
from .prompts import AgentPrompts

__all__ = [
    "MeetingAgent",
    "AgentState",
    "TodoItem",
    "TodoStatus",
    "Suggestion",
    "AgentPrompts",
]
