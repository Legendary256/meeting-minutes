"""
Meeting Agent - Real-time AI assistant for meetings.
Tracks agenda, suggests questions, updates TODO list.
Uses Google Gemini 2.0 Flash for fast inference.
"""
import asyncio
import logging
import os
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TodoStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DISCUSSED = "discussed"
    SKIPPED = "skipped"


@dataclass
class TodoItem:
    """Meeting TODO list item."""
    id: str
    topic: str
    description: str
    status: TodoStatus = TodoStatus.PENDING
    priority: int = 1  # 1 = highest
    notes: List[str] = field(default_factory=list)
    discussed_at: Optional[datetime] = None
    key_points: List[str] = field(default_factory=list)
    
    def mark_discussed(self, key_points: List[str] = None):
        self.status = TodoStatus.DISCUSSED
        self.discussed_at = datetime.now()
        if key_points:
            self.key_points.extend(key_points)


@dataclass
class Suggestion:
    """Agent suggestion."""
    type: str  # "question", "followup", "reminder", "warning"
    content: str
    related_todo_id: Optional[str] = None
    priority: str = "normal"  # "low", "normal", "high", "urgent"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentState:
    """Agent state during meeting."""
    meeting_id: str
    agenda: str
    goals: List[str]
    todos: List[TodoItem] = field(default_factory=list)
    suggestions: List[Suggestion] = field(default_factory=list)
    transcript_chunks: List[str] = field(default_factory=list)
    last_analysis_index: int = 0
    is_active: bool = True
    started_at: datetime = field(default_factory=datetime.now)


class MeetingAgent:
    """
    AI Agent supporting meeting facilitation using Google Gemini.
    
    Features:
    - Generates TODO list from agenda
    - Tracks meeting progress in real-time
    - Suggests questions and follow-ups
    - Updates topic status
    - Warns about skipping important points
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = None,
        analysis_interval: int = None,
        on_update: Optional[Callable[['AgentState'], None]] = None
    ):
        """
        Args:
            api_key: Google API key (or from env GOOGLE_API_KEY)
            model: Gemini model to use (default: gemini-2.0-flash)
            analysis_interval: How often to analyze new fragment (seconds)
            on_update: Callback called on each state update
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model or os.getenv("MEETING_AGENT_MODEL", "gemini-3-flash")
        self.analysis_interval = analysis_interval or int(
            os.getenv("MEETING_AGENT_ANALYSIS_INTERVAL", "45")
        )
        self.on_update = on_update
        self._client = None
        self._gemini_model = None
        self._prompts = None
        
        self._state: Optional[AgentState] = None
        self._analysis_task: Optional[asyncio.Task] = None
    
    @property
    def client(self):
        """Lazy load Google Generative AI client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai
                self._gemini_model = genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError(
                    "google-generativeai package not installed. "
                    "Install it with: pip install google-generativeai"
                )
        return self._client
    
    @property
    def gemini_model(self):
        """Get the configured Gemini model."""
        if self._gemini_model is None:
            # Trigger client initialization
            _ = self.client
        return self._gemini_model
    
    @property
    def prompts(self):
        """Lazy load prompts."""
        if self._prompts is None:
            from .prompts import AgentPrompts
            self._prompts = AgentPrompts()
        return self._prompts
    
    async def start_meeting(
        self,
        meeting_id: str,
        agenda: str,
        goals: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> AgentState:
        """
        Start new meeting.
        
        Args:
            meeting_id: Unique meeting ID
            agenda: Meeting agenda (text or markdown)
            goals: List of goals to achieve
            context: Additional context (e.g., previous meetings)
        
        Returns:
            Initial agent state with generated TODO list
        """
        logger.info(f"Starting meeting agent for: {meeting_id}")
        
        # Initialize state
        self._state = AgentState(
            meeting_id=meeting_id,
            agenda=agenda,
            goals=goals or []
        )
        
        # Generate TODO list from agenda
        todos = await self._generate_initial_todos(agenda, goals, context)
        self._state.todos = todos
        
        # Start background analysis task
        self._analysis_task = asyncio.create_task(
            self._analysis_loop()
        )
        
        if self.on_update:
            self.on_update(self._state)
        
        return self._state
    
    async def add_transcript_chunk(self, text: str):
        """
        Add new transcript fragment for analysis.
        Called by transcription system.
        """
        if not self._state or not self._state.is_active:
            return
        
        self._state.transcript_chunks.append(text)
    
    async def analyze_now(self) -> Dict[str, Any]:
        """
        Force immediate analysis (e.g., on user request).
        """
        if not self._state:
            raise ValueError("No active meeting")
        
        return await self._analyze_transcript()
    
    async def end_meeting(self) -> Dict[str, Any]:
        """
        End meeting and generate summary.
        """
        if not self._state:
            raise ValueError("No active meeting")
        
        self._state.is_active = False
        
        # Cancel background task
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
        
        # Generate final summary
        summary = await self._generate_final_summary()
        
        return summary
    
    def get_state(self) -> Optional[AgentState]:
        """Get current agent state."""
        return self._state
    
    def get_pending_todos(self) -> List[TodoItem]:
        """Get undiscussed topics."""
        if not self._state:
            return []
        return [t for t in self._state.todos if t.status == TodoStatus.PENDING]
    
    def get_recent_suggestions(self, limit: int = 5) -> List[Suggestion]:
        """Get recent suggestions."""
        if not self._state:
            return []
        return sorted(
            self._state.suggestions, 
            key=lambda s: s.timestamp, 
            reverse=True
        )[:limit]
    
    # ============== PRIVATE METHODS ==============
    
    async def _generate_initial_todos(
        self,
        agenda: str,
        goals: Optional[List[str]],
        context: Optional[str]
    ) -> List[TodoItem]:
        """Generate TODO list from agenda using Gemini."""
        
        prompt = self.prompts.generate_todos_prompt(agenda, goals, context)
        
        # Use Gemini API
        response = self.gemini_model.generate_content(prompt)
        
        # Parse response (expecting JSON)
        import json
        try:
            # Extract JSON from response
            content = response.text
            # Find JSON in response
            start = content.find('[')
            end = content.rfind(']') + 1
            if start != -1 and end > start:
                todos_data = json.loads(content[start:end])
            else:
                # Fallback - try entire content
                todos_data = json.loads(content)
            
            return [
                TodoItem(
                    id=f"todo_{i}",
                    topic=item.get("topic", ""),
                    description=item.get("description", ""),
                    priority=item.get("priority", i + 1)
                )
                for i, item in enumerate(todos_data)
            ]
        except json.JSONDecodeError:
            logger.error(f"Failed to parse todos JSON: {content}")
            # Fallback - create one TODO from entire agenda
            return [TodoItem(
                id="todo_0",
                topic="Discuss agenda",
                description=agenda[:200]
            )]
    
    async def _analysis_loop(self):
        """Background loop for periodic analysis."""
        while self._state and self._state.is_active:
            try:
                await asyncio.sleep(self.analysis_interval)
                
                if not self._state.is_active:
                    break
                
                # Check if there's new text to analyze
                if len(self._state.transcript_chunks) > self._state.last_analysis_index:
                    await self._analyze_transcript()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
    
    async def _analyze_transcript(self) -> Dict[str, Any]:
        """Analyze new transcript fragments using Gemini."""
        
        # Get new fragments
        new_chunks = self._state.transcript_chunks[self._state.last_analysis_index:]
        if not new_chunks:
            return {"status": "no_new_content"}
        
        new_text = " ".join(new_chunks)
        self._state.last_analysis_index = len(self._state.transcript_chunks)
        
        # Prepare context
        pending_topics = [t.topic for t in self.get_pending_todos()]
        discussed_topics = [
            t.topic for t in self._state.todos 
            if t.status == TodoStatus.DISCUSSED
        ]
        
        prompt = self.prompts.analyze_transcript_prompt(
            transcript=new_text,
            pending_topics=pending_topics,
            discussed_topics=discussed_topics,
            goals=self._state.goals
        )
        
        # Use Gemini API
        response = self.gemini_model.generate_content(prompt)
        
        # Parse response
        analysis = self._parse_analysis_response(response.text)
        
        # Update state based on analysis
        self._apply_analysis(analysis)
        
        if self.on_update:
            self.on_update(self._state)
        
        return analysis
    
    def _parse_analysis_response(self, content: str) -> Dict[str, Any]:
        """Parse analysis response."""
        import json
        try:
            # Find JSON in response
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass
        
        # Fallback
        return {
            "topics_discussed": [],
            "suggestions": [],
            "warnings": []
        }
    
    def _apply_analysis(self, analysis: Dict[str, Any]):
        """Apply analysis results to state."""
        
        # Update topic status
        for topic_info in analysis.get("topics_discussed", []):
            topic_name = topic_info.get("topic", "")
            key_points = topic_info.get("key_points", [])
            
            for todo in self._state.todos:
                if self._topics_match(todo.topic, topic_name):
                    todo.mark_discussed(key_points)
                    break
        
        # Add new suggestions
        for suggestion_data in analysis.get("suggestions", []):
            suggestion = Suggestion(
                type=suggestion_data.get("type", "question"),
                content=suggestion_data.get("content", ""),
                priority=suggestion_data.get("priority", "normal"),
                related_todo_id=suggestion_data.get("related_todo")
            )
            self._state.suggestions.append(suggestion)
        
        # Add warnings as high-priority suggestions
        for warning in analysis.get("warnings", []):
            suggestion = Suggestion(
                type="warning",
                content=warning,
                priority="high"
            )
            self._state.suggestions.append(suggestion)
    
    def _topics_match(self, topic1: str, topic2: str) -> bool:
        """Check if two topics are similar."""
        # Simple comparison - can be extended with embeddings
        t1 = topic1.lower().strip()
        t2 = topic2.lower().strip()
        
        # Exact match
        if t1 == t2:
            return True
        
        # Partial match
        if t1 in t2 or t2 in t1:
            return True
        
        # Check common keywords
        words1 = set(t1.split())
        words2 = set(t2.split())
        common = words1 & words2
        
        return len(common) >= 2
    
    async def _generate_final_summary(self) -> Dict[str, Any]:
        """Generate final meeting summary using Gemini."""
        
        full_transcript = " ".join(self._state.transcript_chunks)
        
        prompt = self.prompts.final_summary_prompt(
            transcript=full_transcript,
            todos=self._state.todos,
            goals=self._state.goals
        )
        
        # Use Gemini API
        response = self.gemini_model.generate_content(prompt)
        
        return {
            "summary": response.text,
            "todos_completed": [
                {"id": t.id, "topic": t.topic, "key_points": t.key_points}
                for t in self._state.todos 
                if t.status == TodoStatus.DISCUSSED
            ],
            "todos_missed": [
                {"id": t.id, "topic": t.topic, "description": t.description}
                for t in self._state.todos 
                if t.status == TodoStatus.PENDING
            ],
            "total_suggestions": len(self._state.suggestions),
            "duration_minutes": (
                datetime.now() - self._state.started_at
            ).total_seconds() / 60
        }


# Global agent storage (in production use Redis/DB)
_active_agents: Dict[str, MeetingAgent] = {}


def get_active_agent(meeting_id: str) -> Optional[MeetingAgent]:
    """Get active agent for meeting."""
    return _active_agents.get(meeting_id)


def register_agent(meeting_id: str, agent: MeetingAgent):
    """Register agent for meeting."""
    _active_agents[meeting_id] = agent


def unregister_agent(meeting_id: str):
    """Unregister agent for meeting."""
    if meeting_id in _active_agents:
        del _active_agents[meeting_id]
