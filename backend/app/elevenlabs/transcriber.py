"""
ElevenLabs Scribe v2 Transcription Module
"""
import os
import time
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ElevenLabsConfig:
    api_key: str
    model_id: str = "scribe_v2"
    default_language: Optional[str] = None  # None = auto-detect
    enable_diarization: bool = True
    max_speakers: int = 10
    timestamps_granularity: str = "word"  # "word" or "character"
    tag_audio_events: bool = True


class ElevenLabsTranscriber:
    """
    Wrapper for ElevenLabs Scribe v2 API with cost calculation.
    """
    
    # ElevenLabs pricing (per hour of audio)
    PRICING = {
        "free": 0.0,        # Free 30 min/month
        "starter": 0.40,    # $0.40/h
        "creator": 0.35,    # $0.35/h ($22/month plan)
        "pro": 0.33,        # $0.33/h ($99/month plan)
        "scale": 0.30,      # $0.30/h ($330/month plan)
        "enterprise": 0.25  # Custom pricing
    }
    
    def __init__(self, config: ElevenLabsConfig, pricing_tier: str = "pro"):
        self.config = config
        self._client = None
        self._cost_calculator = None
        self.pricing_tier = pricing_tier
        self._session_stats = {
            "total_duration_seconds": 0,
            "total_cost": 0.0,
            "transcriptions_count": 0
        }
    
    @property
    def client(self):
        """Lazy load ElevenLabs client."""
        if self._client is None:
            try:
                from elevenlabs import ElevenLabs
                self._client = ElevenLabs(api_key=self.config.api_key)
            except ImportError:
                raise ImportError(
                    "elevenlabs package not installed. "
                    "Install it with: pip install elevenlabs"
                )
        return self._client
    
    @property
    def cost_calculator(self):
        """Lazy load cost calculator."""
        if self._cost_calculator is None:
            from .cost_calculator import CostCalculator
            self._cost_calculator = CostCalculator(
                price_per_hour=self.PRICING.get(self.pricing_tier, 0.33)
            )
        return self._cost_calculator
    
    async def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = None,
        num_speakers: Optional[int] = None,
        custom_vocabulary: Optional[List[str]] = None
    ):
        """
        Transcribe audio file with diarization and cost calculation.
        
        Args:
            audio_path: Path to audio/video file
            language: ISO-639-1 language code (pl, en, etc.) or None for auto-detect
            num_speakers: Expected number of speakers (1-32) or None for auto
            custom_vocabulary: List of words/phrases for better recognition
        
        Returns:
            TranscriptionResult with text, speakers, and costs
        """
        from .models import TranscriptionResult, Speaker, Word
        
        start_time = time.time()
        
        # Prepare parameters
        params = {
            "model_id": self.config.model_id,
            "diarize": self.config.enable_diarization,
            "timestamps_granularity": self.config.timestamps_granularity,
            "tag_audio_events": self.config.tag_audio_events,
        }
        
        # Language - use provided or default
        effective_language = language or self.config.default_language
        if effective_language:
            params["language_code"] = effective_language
        
        # Number of speakers
        if num_speakers:
            params["num_speakers"] = min(num_speakers, 32)
        elif self.config.max_speakers:
            params["num_speakers"] = self.config.max_speakers
        
        # Custom vocabulary (keyterm prompting in Scribe v2)
        if custom_vocabulary:
            # Scribe v2 supports up to 100 keyterms
            params["keyterms"] = custom_vocabulary[:100]
        
        logger.info(f"Starting transcription: {audio_path}, language={effective_language}")
        
        try:
            # Call API
            with open(audio_path, "rb") as audio_file:
                response = self.client.speech_to_text.convert(
                    file=audio_file,
                    **params
                )
            
            # Get audio duration (from response or file)
            audio_duration_seconds = self._get_audio_duration(response, audio_path)
            
            # Calculate cost
            cost = self.cost_calculator.calculate(audio_duration_seconds)
            
            # Update session stats
            self._session_stats["total_duration_seconds"] += audio_duration_seconds
            self._session_stats["total_cost"] += cost
            self._session_stats["transcriptions_count"] += 1
            
            # Parse result
            result = self._parse_response(response, audio_duration_seconds, cost)
            
            processing_time = time.time() - start_time
            logger.info(
                f"Transcription complete: {audio_duration_seconds:.1f}s audio, "
                f"cost=${cost:.4f}, processing={processing_time:.1f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def _parse_response(
        self, 
        response: Any, 
        duration: float, 
        cost: float
    ):
        """Parse API response to structured result."""
        from .models import TranscriptionResult, Speaker, Word
        
        # Extract words with timestamps and speakers
        words = []
        speakers = {}
        
        for word_data in getattr(response, 'words', []):
            word = Word(
                text=word_data.text,
                start=word_data.start,
                end=word_data.end,
                speaker=getattr(word_data, 'speaker', None),
                confidence=getattr(word_data, 'confidence', None)
            )
            words.append(word)
            
            # Collect unique speaker IDs
            if word.speaker and word.speaker not in speakers:
                speakers[word.speaker] = Speaker(
                    id=word.speaker,
                    name=f"Speaker {len(speakers) + 1}"
                )
        
        # Build full text with speaker labels
        full_text = self._build_transcript_text(words)
        
        # Detect language (if auto-detect)
        detected_language = getattr(response, 'language', None)
        
        return TranscriptionResult(
            text=response.text,
            formatted_text=full_text,
            words=words,
            speakers=list(speakers.values()),
            language=detected_language,
            duration_seconds=duration,
            cost_usd=cost,
            model=self.config.model_id,
            audio_events=getattr(response, 'audio_events', [])
        )
    
    def _build_transcript_text(self, words: List) -> str:
        """Build formatted text with speaker labels."""
        if not words:
            return ""
        
        lines = []
        current_speaker = None
        current_line = []
        
        for word in words:
            if word.speaker != current_speaker:
                # Save previous line
                if current_line:
                    speaker_label = f"[{current_speaker}]" if current_speaker else ""
                    lines.append(f"{speaker_label} {' '.join(current_line)}")
                
                current_speaker = word.speaker
                current_line = [word.text]
            else:
                current_line.append(word.text)
        
        # Last line
        if current_line:
            speaker_label = f"[{current_speaker}]" if current_speaker else ""
            lines.append(f"{speaker_label} {' '.join(current_line)}")
        
        return "\n\n".join(lines)
    
    def _get_audio_duration(self, response: Any, audio_path: str) -> float:
        """Get audio duration from response or file."""
        # Try from response
        if hasattr(response, 'duration'):
            return response.duration
        
        # Try from last word
        if hasattr(response, 'words') and response.words:
            last_word = response.words[-1]
            if hasattr(last_word, 'end'):
                return last_word.end
        
        # Fallback: use ffprobe
        try:
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 
                 'format=duration', '-of', 'csv=p=0', audio_path],
                capture_output=True, text=True
            )
            return float(result.stdout.strip())
        except:
            return 0.0
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Return session statistics."""
        return {
            **self._session_stats,
            "total_duration_formatted": self._format_duration(
                self._session_stats["total_duration_seconds"]
            ),
            "total_cost_formatted": f"${self._session_stats['total_cost']:.2f}"
        }
    
    def reset_session_stats(self):
        """Reset session statistics."""
        self._session_stats = {
            "total_duration_seconds": 0,
            "total_cost": 0.0,
            "transcriptions_count": 0
        }
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds to HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# Singleton for easy access
_transcriber_instance: Optional[ElevenLabsTranscriber] = None


def get_transcriber() -> ElevenLabsTranscriber:
    """Get transcriber instance (singleton)."""
    global _transcriber_instance
    if _transcriber_instance is None:
        api_key = os.getenv("ELEVENLABS_API_KEY", "")
        if not api_key:
            raise ValueError(
                "ELEVENLABS_API_KEY environment variable not set. "
                "Please set it in your .env file."
            )
        
        config = ElevenLabsConfig(
            api_key=api_key,
            model_id=os.getenv("ELEVENLABS_MODEL", "scribe_v2"),
            default_language=os.getenv("ELEVENLABS_DEFAULT_LANGUAGE") or None,
            enable_diarization=True
        )
        _transcriber_instance = ElevenLabsTranscriber(
            config=config,
            pricing_tier=os.getenv("ELEVENLABS_PRICING_TIER", "pro")
        )
    return _transcriber_instance
