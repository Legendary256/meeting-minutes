"""
ElevenLabs Scribe v2 Transcription Module

This module provides integration with ElevenLabs Scribe v2 API for
high-quality speech-to-text transcription with speaker diarization.
"""

from .transcriber import ElevenLabsTranscriber, ElevenLabsConfig, get_transcriber
from .cost_calculator import CostCalculator, CostSummary
from .models import (
    TranscriptionResult,
    TranscriptionRequest,
    Word,
    Speaker,
    AudioEvent,
    CostStatsResponse,
    SUPPORTED_LANGUAGES,
)

__all__ = [
    "ElevenLabsTranscriber",
    "ElevenLabsConfig",
    "get_transcriber",
    "CostCalculator",
    "CostSummary",
    "TranscriptionResult",
    "TranscriptionRequest",
    "Word",
    "Speaker",
    "AudioEvent",
    "CostStatsResponse",
    "SUPPORTED_LANGUAGES",
]
