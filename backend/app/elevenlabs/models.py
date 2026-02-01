"""
Pydantic models for ElevenLabs module
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime


class Word(BaseModel):
    """Single word with timestamp."""
    text: str
    start: float  # seconds
    end: float
    speaker: Optional[str] = None
    confidence: Optional[float] = None


class Speaker(BaseModel):
    """Speaker information."""
    id: str
    name: str = "Unknown"
    # Optional: mapping to real names
    real_name: Optional[str] = None


class AudioEvent(BaseModel):
    """Audio event (laughter, applause, etc.)."""
    type: str
    start: float
    end: float


class TranscriptionResult(BaseModel):
    """Complete transcription result."""
    text: str  # Raw text
    formatted_text: str  # Text with speaker labels
    words: List[Word]
    speakers: List[Speaker]
    language: Optional[str] = None
    duration_seconds: float
    cost_usd: float
    model: str
    audio_events: List[Any] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TranscriptionRequest(BaseModel):
    """Request for transcription."""
    audio_path: str
    language: Optional[str] = None  # None = auto-detect
    num_speakers: Optional[int] = None
    custom_vocabulary: Optional[List[str]] = None
    meeting_id: Optional[str] = None


class CostStatsResponse(BaseModel):
    """Cost statistics response."""
    total_cost_usd: float
    total_duration_seconds: float
    total_duration_formatted: str
    transcription_count: int
    current_month_cost: float
    current_month_hours: float


class LanguageOption(BaseModel):
    """Language option for UI."""
    code: str
    name: str
    native_name: str


# Supported languages (subset, full list has 90+)
SUPPORTED_LANGUAGES: List[LanguageOption] = [
    LanguageOption(code="pl", name="Polish", native_name="Polski"),
    LanguageOption(code="en", name="English", native_name="English"),
    LanguageOption(code="de", name="German", native_name="Deutsch"),
    LanguageOption(code="fr", name="French", native_name="Français"),
    LanguageOption(code="es", name="Spanish", native_name="Español"),
    LanguageOption(code="it", name="Italian", native_name="Italiano"),
    LanguageOption(code="pt", name="Portuguese", native_name="Português"),
    LanguageOption(code="nl", name="Dutch", native_name="Nederlands"),
    LanguageOption(code="uk", name="Ukrainian", native_name="Українська"),
    LanguageOption(code="cs", name="Czech", native_name="Čeština"),
    LanguageOption(code="ja", name="Japanese", native_name="日本語"),
    LanguageOption(code="zh", name="Chinese", native_name="中文"),
    LanguageOption(code="ko", name="Korean", native_name="한국어"),
    LanguageOption(code="ru", name="Russian", native_name="Русский"),
    LanguageOption(code="ar", name="Arabic", native_name="العربية"),
]
