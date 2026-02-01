# 🎯 Meetily + ElevenLabs Scribe v2 + Meeting Agent AI

## Plan implementacji "od zera do działającego produktu"

---

## 📋 Spis treści

1. [Architektura obecnego Meetily](#1-architektura-obecnego-meetily)
2. [Faza 1: Fork i setup środowiska](#2-faza-1-fork-i-setup-środowiska)
3. [Faza 2: Moduł ElevenLabs](#3-faza-2-moduł-elevenlabs)
4. [Faza 3: Meeting Agent AI](#4-faza-3-meeting-agent-ai)
5. [Faza 4: Aplikacja mobilna iOS](#5-faza-4-aplikacja-mobilna-ios)
6. [Szczegóły implementacji](#6-szczegóły-implementacji)
7. [Koszty i cennik](#7-koszty-i-cennik)

---

## 1. Architektura obecnego Meetily

### Stack technologiczny

```
┌─────────────────────────────────────────────────────────────┐
│                    MEETILY ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│  FRONTEND (Tauri + Next.js)           Port: 3118 (dev)      │
│  ├── React/TypeScript UI                                     │
│  ├── Tauri IPC → Rust backend                               │
│  └── WebSocket → FastAPI                                     │
├─────────────────────────────────────────────────────────────┤
│  RUST BACKEND (w Tauri)                                      │
│  ├── Audio capture (mic + system)                           │
│  ├── 6-stage audio pipeline                                  │
│  │   └── mono→HPF→resample 48kHz→denoise→normalize→16kHz    │
│  └── IPC commands do frontend                               │
├─────────────────────────────────────────────────────────────┤
│  FASTAPI BACKEND                      Port: 5167            │
│  ├── /process-transcript → LLM summarization                │
│  ├── /meetings/* → CRUD operations                          │
│  ├── Integracje: Claude, Groq, OpenRouter, Ollama           │
│  └── SQLite database                                         │
├─────────────────────────────────────────────────────────────┤
│  WHISPER SERVER                       Port: 8178            │
│  ├── whisper.cpp / Parakeet                                 │
│  └── Real-time transcription                                │
├─────────────────────────────────────────────────────────────┤
│  DATABASE                                                    │
│  ├── meeting_minutes.db (SQLite)                            │
│  └── ChromaDB (vector store, opcjonalnie)                   │
└─────────────────────────────────────────────────────────────┘
```

### Kluczowe pliki do modyfikacji

```
meeting-minutes/
├── backend/
│   ├── app/
│   │   ├── main.py              ← FastAPI endpoints (MODYFIKACJA)
│   │   ├── transcript_processor.py  ← LLM processing
│   │   └── database_manager.py  ← SQLite operations
│   └── requirements.txt         ← Python dependencies (MODYFIKACJA)
│
├── frontend/
│   ├── src/
│   │   ├── app/                 ← Next.js pages
│   │   ├── components/          ← React components (MODYFIKACJA)
│   │   └── lib/                 ← Utilities
│   ├── src-tauri/
│   │   ├── src/
│   │   │   ├── main.rs          ← Tauri entry point
│   │   │   └── audio/           ← Audio capture (Rust)
│   │   └── Cargo.toml           ← Rust dependencies
│   └── package.json             ← Node dependencies
│
└── docs/                        ← Documentation
```

---

## 2. Faza 1: Fork i setup środowiska

### Krok 1.1: Fork repozytorium

```bash
# 1. Fork na GitHubie: https://github.com/Zackriya-Solutions/meeting-minutes
#    Kliknij "Fork" → wybierz swoje konto

# 2. Sklonuj swój fork
git clone https://github.com/TWOJ_USERNAME/meeting-minutes.git
cd meeting-minutes

# 3. Dodaj upstream (oryginalne repo)
git remote add upstream https://github.com/Zackriya-Solutions/meeting-minutes.git

# 4. Utwórz branch dla swoich zmian
git checkout -b feature/elevenlabs-agent
```

### Krok 1.2: Setup środowiska (macOS)

```bash
# Wymagania systemowe
xcode-select --install  # Xcode CLI tools
brew install rust node pnpm python@3.11 ffmpeg

# Backend Python
cd backend
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install elevenlabs anthropic  # Nowe zależności

# Frontend
cd ../frontend
pnpm install

# Test czy działa oryginał
pnpm run tauri:dev
```

### Krok 1.3: Struktura nowych modułów

```bash
# Utwórz strukturę dla nowych funkcji
mkdir -p backend/app/elevenlabs
mkdir -p backend/app/agent
mkdir -p frontend/src/components/agent

touch backend/app/elevenlabs/__init__.py
touch backend/app/elevenlabs/transcriber.py
touch backend/app/elevenlabs/cost_calculator.py
touch backend/app/agent/__init__.py
touch backend/app/agent/meeting_agent.py
touch backend/app/agent/prompts.py
```

---

## 3. Faza 2: Moduł ElevenLabs

### 3.1 Architektura modułu ElevenLabs

```
backend/app/elevenlabs/
├── __init__.py
├── transcriber.py       ← Główna klasa transkrypcji
├── cost_calculator.py   ← Kalkulacja kosztów API
├── models.py            ← Pydantic schemas
└── config.py            ← Konfiguracja (API key, modele)
```

### 3.2 Implementacja: `transcriber.py`

```python
# backend/app/elevenlabs/transcriber.py
"""
ElevenLabs Scribe v2 Transcription Module
"""
import os
import time
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from elevenlabs import ElevenLabs
from .cost_calculator import CostCalculator
from .models import TranscriptionResult, Speaker, Word

logger = logging.getLogger(__name__)


@dataclass
class ElevenLabsConfig:
    api_key: str
    model_id: str = "scribe_v2"  # lub "scribe_v2" gdy dostępny
    default_language: Optional[str] = None  # None = auto-detect
    enable_diarization: bool = True
    max_speakers: int = 10
    timestamps_granularity: str = "word"  # "word" lub "character"
    tag_audio_events: bool = True


class ElevenLabsTranscriber:
    """
    Wrapper dla ElevenLabs Scribe v2 API z kalkulacją kosztów.
    """
    
    # Cennik ElevenLabs (per hour of audio)
    PRICING = {
        "free": 0.0,        # Darmowe 30 min/mies
        "starter": 0.40,    # $0.40/h
        "creator": 0.35,    # $0.35/h (plan $22/mies)
        "pro": 0.33,        # $0.33/h (plan $99/mies)
        "scale": 0.30,      # $0.30/h (plan $330/mies)
        "enterprise": 0.25  # Custom pricing
    }
    
    def __init__(self, config: ElevenLabsConfig, pricing_tier: str = "pro"):
        self.config = config
        self.client = ElevenLabs(api_key=config.api_key)
        self.cost_calculator = CostCalculator(
            price_per_hour=self.PRICING.get(pricing_tier, 0.33)
        )
        self._session_stats = {
            "total_duration_seconds": 0,
            "total_cost": 0.0,
            "transcriptions_count": 0
        }
    
    async def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = None,
        num_speakers: Optional[int] = None,
        custom_vocabulary: Optional[List[str]] = None
    ) -> TranscriptionResult:
        """
        Transkrybuj plik audio z diaryzacją i kalkulacją kosztów.
        
        Args:
            audio_path: Ścieżka do pliku audio/video
            language: Kod języka ISO-639-1 (pl, en, etc.) lub None dla auto-detect
            num_speakers: Oczekiwana liczba mówców (1-32) lub None dla auto
            custom_vocabulary: Lista słów/fraz do lepszego rozpoznawania
        
        Returns:
            TranscriptionResult z tekstem, mówcami i kosztami
        """
        start_time = time.time()
        
        # Przygotuj parametry
        params = {
            "model_id": self.config.model_id,
            "diarize": self.config.enable_diarization,
            "timestamps_granularity": self.config.timestamps_granularity,
            "tag_audio_events": self.config.tag_audio_events,
        }
        
        # Język - użyj podanego lub domyślnego
        effective_language = language or self.config.default_language
        if effective_language:
            params["language_code"] = effective_language
        
        # Liczba mówców
        if num_speakers:
            params["num_speakers"] = min(num_speakers, 32)
        elif self.config.max_speakers:
            params["num_speakers"] = self.config.max_speakers
        
        # Custom vocabulary (keyterm prompting w Scribe v2)
        if custom_vocabulary:
            # Scribe v2 obsługuje do 100 keyterms
            params["keyterms"] = custom_vocabulary[:100]
        
        logger.info(f"Starting transcription: {audio_path}, language={effective_language}")
        
        try:
            # Wywołaj API
            with open(audio_path, "rb") as audio_file:
                response = self.client.speech_to_text.convert(
                    file=audio_file,
                    **params
                )
            
            # Oblicz czas trwania audio (z response lub z pliku)
            audio_duration_seconds = self._get_audio_duration(response, audio_path)
            
            # Oblicz koszt
            cost = self.cost_calculator.calculate(audio_duration_seconds)
            
            # Aktualizuj statystyki sesji
            self._session_stats["total_duration_seconds"] += audio_duration_seconds
            self._session_stats["total_cost"] += cost
            self._session_stats["transcriptions_count"] += 1
            
            # Parsuj wynik
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
    ) -> TranscriptionResult:
        """Parsuj odpowiedź API do ustrukturyzowanego wyniku."""
        
        # Wyciągnij słowa z timestampami i mówcami
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
            
            # Zbierz unikalne speaker IDs
            if word.speaker and word.speaker not in speakers:
                speakers[word.speaker] = Speaker(
                    id=word.speaker,
                    name=f"Mówca {len(speakers) + 1}"  # Domyślna nazwa
                )
        
        # Zbuduj pełny tekst z oznaczeniem mówców
        full_text = self._build_transcript_text(words)
        
        # Wykryj język (jeśli auto-detect)
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
    
    def _build_transcript_text(self, words: List[Word]) -> str:
        """Zbuduj sformatowany tekst z podziałem na mówców."""
        if not words:
            return ""
        
        lines = []
        current_speaker = None
        current_line = []
        
        for word in words:
            if word.speaker != current_speaker:
                # Zapisz poprzednią linię
                if current_line:
                    speaker_label = f"[{current_speaker}]" if current_speaker else ""
                    lines.append(f"{speaker_label} {' '.join(current_line)}")
                
                current_speaker = word.speaker
                current_line = [word.text]
            else:
                current_line.append(word.text)
        
        # Ostatnia linia
        if current_line:
            speaker_label = f"[{current_speaker}]" if current_speaker else ""
            lines.append(f"{speaker_label} {' '.join(current_line)}")
        
        return "\n\n".join(lines)
    
    def _get_audio_duration(self, response: Any, audio_path: str) -> float:
        """Pobierz czas trwania audio z response lub z pliku."""
        # Próbuj z response
        if hasattr(response, 'duration'):
            return response.duration
        
        # Próbuj z ostatniego słowa
        if hasattr(response, 'words') and response.words:
            last_word = response.words[-1]
            if hasattr(last_word, 'end'):
                return last_word.end
        
        # Fallback: użyj ffprobe
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
        """Zwróć statystyki bieżącej sesji."""
        return {
            **self._session_stats,
            "total_duration_formatted": self._format_duration(
                self._session_stats["total_duration_seconds"]
            ),
            "total_cost_formatted": f"${self._session_stats['total_cost']:.2f}"
        }
    
    def reset_session_stats(self):
        """Zresetuj statystyki sesji."""
        self._session_stats = {
            "total_duration_seconds": 0,
            "total_cost": 0.0,
            "transcriptions_count": 0
        }
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Formatuj sekundy do HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# Singleton dla łatwego dostępu
_transcriber_instance: Optional[ElevenLabsTranscriber] = None


def get_transcriber() -> ElevenLabsTranscriber:
    """Pobierz instancję transcribera (singleton)."""
    global _transcriber_instance
    if _transcriber_instance is None:
        config = ElevenLabsConfig(
            api_key=os.getenv("ELEVENLABS_API_KEY", ""),
            model_id=os.getenv("ELEVENLABS_MODEL", "scribe_v2"),
            default_language=os.getenv("ELEVENLABS_DEFAULT_LANGUAGE"),
            enable_diarization=True
        )
        _transcriber_instance = ElevenLabsTranscriber(
            config=config,
            pricing_tier=os.getenv("ELEVENLABS_PRICING_TIER", "pro")
        )
    return _transcriber_instance
```

### 3.3 Implementacja: `cost_calculator.py`

```python
# backend/app/elevenlabs/cost_calculator.py
"""
Kalkulator kosztów ElevenLabs API
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, date
import json
import os


@dataclass
class TranscriptionCostEntry:
    """Pojedynczy wpis kosztowy."""
    timestamp: datetime
    duration_seconds: float
    cost_usd: float
    meeting_id: Optional[str] = None
    meeting_name: Optional[str] = None


@dataclass
class CostSummary:
    """Podsumowanie kosztów."""
    total_cost_usd: float
    total_duration_seconds: float
    transcription_count: int
    period_start: Optional[date] = None
    period_end: Optional[date] = None
    
    @property
    def total_duration_hours(self) -> float:
        return self.total_duration_seconds / 3600
    
    @property
    def average_cost_per_hour(self) -> float:
        if self.total_duration_hours == 0:
            return 0
        return self.total_cost_usd / self.total_duration_hours


class CostCalculator:
    """
    Kalkulator i tracker kosztów transkrypcji ElevenLabs.
    """
    
    def __init__(
        self, 
        price_per_hour: float = 0.33,
        history_file: Optional[str] = None
    ):
        """
        Args:
            price_per_hour: Cena za godzinę audio w USD
            history_file: Ścieżka do pliku z historią kosztów (JSON)
        """
        self.price_per_hour = price_per_hour
        self.price_per_second = price_per_hour / 3600
        self.history_file = history_file or os.path.expanduser(
            "~/.meetily/elevenlabs_costs.json"
        )
        self._history: List[TranscriptionCostEntry] = []
        self._load_history()
    
    def calculate(self, duration_seconds: float) -> float:
        """
        Oblicz koszt dla danego czasu trwania audio.
        
        Args:
            duration_seconds: Czas trwania audio w sekundach
            
        Returns:
            Koszt w USD
        """
        return duration_seconds * self.price_per_second
    
    def log_transcription(
        self,
        duration_seconds: float,
        cost_usd: float,
        meeting_id: Optional[str] = None,
        meeting_name: Optional[str] = None
    ):
        """Zapisz transkrypcję do historii."""
        entry = TranscriptionCostEntry(
            timestamp=datetime.now(),
            duration_seconds=duration_seconds,
            cost_usd=cost_usd,
            meeting_id=meeting_id,
            meeting_name=meeting_name
        )
        self._history.append(entry)
        self._save_history()
    
    def get_summary(
        self,
        period_start: Optional[date] = None,
        period_end: Optional[date] = None
    ) -> CostSummary:
        """
        Pobierz podsumowanie kosztów za okres.
        
        Args:
            period_start: Początek okresu (None = od początku)
            period_end: Koniec okresu (None = do teraz)
        """
        entries = self._filter_by_period(period_start, period_end)
        
        return CostSummary(
            total_cost_usd=sum(e.cost_usd for e in entries),
            total_duration_seconds=sum(e.duration_seconds for e in entries),
            transcription_count=len(entries),
            period_start=period_start,
            period_end=period_end
        )
    
    def get_monthly_summary(self, year: int, month: int) -> CostSummary:
        """Pobierz podsumowanie za konkretny miesiąc."""
        from calendar import monthrange
        
        start = date(year, month, 1)
        _, last_day = monthrange(year, month)
        end = date(year, month, last_day)
        
        return self.get_summary(start, end)
    
    def get_current_month_summary(self) -> CostSummary:
        """Pobierz podsumowanie za bieżący miesiąc."""
        today = date.today()
        return self.get_monthly_summary(today.year, today.month)
    
    def estimate_monthly_cost(self, hours_per_month: float) -> float:
        """Oszacuj miesięczny koszt dla danej liczby godzin."""
        return hours_per_month * self.price_per_hour
    
    def get_plan_recommendation(self, monthly_hours: float) -> Dict:
        """
        Rekomenduj plan ElevenLabs na podstawie miesięcznego użycia.
        """
        plans = [
            {"name": "Free", "price": 0, "hours": 0.5, "per_hour": 0},
            {"name": "Starter ($5)", "price": 5, "hours": 12.5, "per_hour": 0.40},
            {"name": "Creator ($22)", "price": 22, "hours": 63, "per_hour": 0.35},
            {"name": "Pro ($99)", "price": 99, "hours": 300, "per_hour": 0.33},
            {"name": "Scale ($330)", "price": 330, "hours": 1100, "per_hour": 0.30},
        ]
        
        for plan in plans:
            if monthly_hours <= plan["hours"]:
                overage = 0
                total_cost = plan["price"]
                break
        else:
            # Przekracza Scale
            plan = plans[-1]
            overage_hours = monthly_hours - plan["hours"]
            overage = overage_hours * plan["per_hour"]
            total_cost = plan["price"] + overage
        
        return {
            "recommended_plan": plan["name"],
            "plan_price": plan["price"],
            "included_hours": plan["hours"],
            "your_hours": monthly_hours,
            "overage_cost": overage,
            "total_estimated_cost": total_cost
        }
    
    def _filter_by_period(
        self,
        start: Optional[date],
        end: Optional[date]
    ) -> List[TranscriptionCostEntry]:
        """Filtruj historię po okresie."""
        entries = self._history
        
        if start:
            entries = [e for e in entries if e.timestamp.date() >= start]
        if end:
            entries = [e for e in entries if e.timestamp.date() <= end]
        
        return entries
    
    def _load_history(self):
        """Wczytaj historię z pliku."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self._history = [
                        TranscriptionCostEntry(
                            timestamp=datetime.fromisoformat(e['timestamp']),
                            duration_seconds=e['duration_seconds'],
                            cost_usd=e['cost_usd'],
                            meeting_id=e.get('meeting_id'),
                            meeting_name=e.get('meeting_name')
                        )
                        for e in data
                    ]
            except Exception as e:
                print(f"Warning: Could not load cost history: {e}")
                self._history = []
    
    def _save_history(self):
        """Zapisz historię do pliku."""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        
        data = [
            {
                'timestamp': e.timestamp.isoformat(),
                'duration_seconds': e.duration_seconds,
                'cost_usd': e.cost_usd,
                'meeting_id': e.meeting_id,
                'meeting_name': e.meeting_name
            }
            for e in self._history
        ]
        
        with open(self.history_file, 'w') as f:
            json.dump(data, f, indent=2)
```

### 3.4 Implementacja: `models.py`

```python
# backend/app/elevenlabs/models.py
"""
Pydantic models dla ElevenLabs module
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime


class Word(BaseModel):
    """Pojedyncze słowo z timestampem."""
    text: str
    start: float  # sekundy
    end: float
    speaker: Optional[str] = None
    confidence: Optional[float] = None


class Speaker(BaseModel):
    """Informacje o mówcy."""
    id: str
    name: str = "Unknown"
    # Opcjonalnie: możesz dodać mapowanie na prawdziwe imiona
    real_name: Optional[str] = None


class AudioEvent(BaseModel):
    """Zdarzenie audio (śmiech, brawa, etc.)."""
    type: str
    start: float
    end: float


class TranscriptionResult(BaseModel):
    """Pełny wynik transkrypcji."""
    text: str  # Surowy tekst
    formatted_text: str  # Tekst z oznaczeniem mówców
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
    """Request do transkrypcji."""
    audio_path: str
    language: Optional[str] = None  # None = auto-detect
    num_speakers: Optional[int] = None
    custom_vocabulary: Optional[List[str]] = None
    meeting_id: Optional[str] = None


class CostStatsResponse(BaseModel):
    """Statystyki kosztów."""
    total_cost_usd: float
    total_duration_seconds: float
    total_duration_formatted: str
    transcription_count: int
    current_month_cost: float
    current_month_hours: float


class LanguageOption(BaseModel):
    """Opcja języka dla UI."""
    code: str
    name: str
    native_name: str


# Wspierane języki (subset, pełna lista ma 90+)
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
    # Dodaj więcej według potrzeb...
]
```

### 3.5 Integracja z FastAPI: Nowe endpointy

```python
# Dodaj do backend/app/main.py

from fastapi import HTTPException
from app.elevenlabs.transcriber import get_transcriber, ElevenLabsConfig, ElevenLabsTranscriber
from app.elevenlabs.models import (
    TranscriptionRequest, TranscriptionResult, 
    CostStatsResponse, SUPPORTED_LANGUAGES
)
from app.elevenlabs.cost_calculator import CostCalculator

# ============== ELEVENLABS ENDPOINTS ==============

@app.post("/api/elevenlabs/transcribe", response_model=TranscriptionResult)
async def transcribe_with_elevenlabs(request: TranscriptionRequest):
    """
    Transkrybuj audio używając ElevenLabs Scribe v2.
    """
    try:
        transcriber = get_transcriber()
        result = await transcriber.transcribe_file(
            audio_path=request.audio_path,
            language=request.language,
            num_speakers=request.num_speakers,
            custom_vocabulary=request.custom_vocabulary
        )
        
        # Zapisz koszt do historii
        transcriber.cost_calculator.log_transcription(
            duration_seconds=result.duration_seconds,
            cost_usd=result.cost_usd,
            meeting_id=request.meeting_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"ElevenLabs transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/elevenlabs/costs", response_model=CostStatsResponse)
async def get_elevenlabs_costs():
    """Pobierz statystyki kosztów ElevenLabs."""
    transcriber = get_transcriber()
    session_stats = transcriber.get_session_stats()
    monthly = transcriber.cost_calculator.get_current_month_summary()
    
    return CostStatsResponse(
        total_cost_usd=session_stats["total_cost"],
        total_duration_seconds=session_stats["total_duration_seconds"],
        total_duration_formatted=session_stats["total_duration_formatted"],
        transcription_count=session_stats["transcriptions_count"],
        current_month_cost=monthly.total_cost_usd,
        current_month_hours=monthly.total_duration_hours
    )


@app.get("/api/elevenlabs/languages")
async def get_supported_languages():
    """Pobierz listę wspieranych języków."""
    return SUPPORTED_LANGUAGES


@app.post("/api/elevenlabs/estimate-cost")
async def estimate_transcription_cost(duration_seconds: float):
    """Oszacuj koszt transkrypcji przed wykonaniem."""
    transcriber = get_transcriber()
    cost = transcriber.cost_calculator.calculate(duration_seconds)
    return {
        "duration_seconds": duration_seconds,
        "duration_formatted": ElevenLabsTranscriber._format_duration(duration_seconds),
        "estimated_cost_usd": cost,
        "price_per_hour": transcriber.cost_calculator.price_per_hour
    }


@app.get("/api/elevenlabs/plan-recommendation")
async def get_plan_recommendation(monthly_hours: float):
    """Pobierz rekomendację planu na podstawie użycia."""
    transcriber = get_transcriber()
    return transcriber.cost_calculator.get_plan_recommendation(monthly_hours)
```

---

## 4. Faza 3: Meeting Agent AI

### 4.1 Architektura Meeting Agent

```
┌─────────────────────────────────────────────────────────────┐
│                    MEETING AGENT FLOW                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [START MEETING]                                             │
│        │                                                     │
│        ▼                                                     │
│  ┌─────────────┐    Agenda/Cele     ┌──────────────────┐   │
│  │   User      │ ─────────────────► │  Meeting Agent   │   │
│  │   Input     │                    │  (Claude API)    │   │
│  └─────────────┘                    └────────┬─────────┘   │
│                                              │              │
│                                              ▼              │
│                                     ┌──────────────────┐   │
│                                     │  Initial TODO    │   │
│                                     │  List Generated  │   │
│                                     └────────┬─────────┘   │
│                                              │              │
│  ┌──────────────────────────────────────────┼──────────┐   │
│  │              REAL-TIME LOOP              │          │   │
│  │                                          ▼          │   │
│  │  [Transcript]  ──►  [Agent Analysis]  ──►  [UI]    │   │
│  │       │                    │                │       │   │
│  │       │              ┌─────┴─────┐          │       │   │
│  │       │              │           │          │       │   │
│  │       │              ▼           ▼          │       │   │
│  │       │         [Update]    [Suggest]       │       │   │
│  │       │          TODO       Questions       │       │   │
│  │       │                                     │       │   │
│  │  Every 30-60 seconds lub na żądanie        │       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  [END MEETING]                                               │
│        │                                                     │
│        ▼                                                     │
│  ┌─────────────────┐                                        │
│  │  Final Summary  │                                        │
│  │  + Action Items │                                        │
│  └─────────────────┘                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Implementacja: `meeting_agent.py`

```python
# backend/app/agent/meeting_agent.py
"""
Meeting Agent - Real-time AI assistant for meetings.
Tracks agenda, suggests questions, updates TODO list.
"""
import asyncio
import logging
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import anthropic
from .prompts import AgentPrompts

logger = logging.getLogger(__name__)


class TodoStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DISCUSSED = "discussed"
    SKIPPED = "skipped"


@dataclass
class TodoItem:
    """Element TODO listy spotkania."""
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
    """Sugestia od agenta."""
    type: str  # "question", "followup", "reminder", "warning"
    content: str
    related_todo_id: Optional[str] = None
    priority: str = "normal"  # "low", "normal", "high", "urgent"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentState:
    """Stan agenta w trakcie spotkania."""
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
    AI Agent wspomagający prowadzenie spotkań.
    
    Funkcje:
    - Generuje TODO listę z agendy
    - Śledzi postęp spotkania w czasie rzeczywistym
    - Sugeruje pytania i follow-upy
    - Aktualizuje status tematów
    - Ostrzega przed pominięciem ważnych punktów
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        analysis_interval: int = 45,  # sekundy
        on_update: Optional[Callable[[AgentState], None]] = None
    ):
        """
        Args:
            api_key: Anthropic API key (lub z env ANTHROPIC_API_KEY)
            model: Model Claude do użycia
            analysis_interval: Jak często analizować nowy fragment (sekundy)
            on_update: Callback wywoływany przy każdej aktualizacji stanu
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.analysis_interval = analysis_interval
        self.on_update = on_update
        self.prompts = AgentPrompts()
        
        self._state: Optional[AgentState] = None
        self._analysis_task: Optional[asyncio.Task] = None
    
    async def start_meeting(
        self,
        meeting_id: str,
        agenda: str,
        goals: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> AgentState:
        """
        Rozpocznij nowe spotkanie.
        
        Args:
            meeting_id: Unikalny ID spotkania
            agenda: Agenda spotkania (tekst lub markdown)
            goals: Lista celów do osiągnięcia
            context: Dodatkowy kontekst (np. poprzednie spotkania)
        
        Returns:
            Początkowy stan agenta z wygenerowaną TODO listą
        """
        logger.info(f"Starting meeting agent for: {meeting_id}")
        
        # Inicjalizuj stan
        self._state = AgentState(
            meeting_id=meeting_id,
            agenda=agenda,
            goals=goals or []
        )
        
        # Wygeneruj TODO listę z agendy
        todos = await self._generate_initial_todos(agenda, goals, context)
        self._state.todos = todos
        
        # Uruchom background task do analizy
        self._analysis_task = asyncio.create_task(
            self._analysis_loop()
        )
        
        if self.on_update:
            self.on_update(self._state)
        
        return self._state
    
    async def add_transcript_chunk(self, text: str):
        """
        Dodaj nowy fragment transkrypcji do analizy.
        Wywoływane przez system transkrypcji.
        """
        if not self._state or not self._state.is_active:
            return
        
        self._state.transcript_chunks.append(text)
    
    async def analyze_now(self) -> Dict[str, Any]:
        """
        Wymuś natychmiastową analizę (np. na żądanie użytkownika).
        """
        if not self._state:
            raise ValueError("No active meeting")
        
        return await self._analyze_transcript()
    
    async def end_meeting(self) -> Dict[str, Any]:
        """
        Zakończ spotkanie i wygeneruj podsumowanie.
        """
        if not self._state:
            raise ValueError("No active meeting")
        
        self._state.is_active = False
        
        # Anuluj background task
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
        
        # Wygeneruj finalne podsumowanie
        summary = await self._generate_final_summary()
        
        return summary
    
    def get_state(self) -> Optional[AgentState]:
        """Pobierz aktualny stan agenta."""
        return self._state
    
    def get_pending_todos(self) -> List[TodoItem]:
        """Pobierz nieomówione tematy."""
        if not self._state:
            return []
        return [t for t in self._state.todos if t.status == TodoStatus.PENDING]
    
    def get_recent_suggestions(self, limit: int = 5) -> List[Suggestion]:
        """Pobierz ostatnie sugestie."""
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
        """Wygeneruj TODO listę z agendy."""
        
        prompt = self.prompts.generate_todos_prompt(agenda, goals, context)
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parsuj odpowiedź (oczekujemy JSON)
        import json
        try:
            # Wyciągnij JSON z odpowiedzi
            content = response.content[0].text
            # Znajdź JSON w odpowiedzi
            start = content.find('[')
            end = content.rfind(']') + 1
            if start != -1 and end > start:
                todos_data = json.loads(content[start:end])
            else:
                # Fallback - spróbuj całość
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
            # Fallback - stwórz jeden TODO z całej agendy
            return [TodoItem(
                id="todo_0",
                topic="Omów agendę",
                description=agenda[:200]
            )]
    
    async def _analysis_loop(self):
        """Background loop do cyklicznej analizy."""
        while self._state and self._state.is_active:
            try:
                await asyncio.sleep(self.analysis_interval)
                
                if not self._state.is_active:
                    break
                
                # Sprawdź czy jest nowy tekst do analizy
                if len(self._state.transcript_chunks) > self._state.last_analysis_index:
                    await self._analyze_transcript()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
    
    async def _analyze_transcript(self) -> Dict[str, Any]:
        """Analizuj nowe fragmenty transkrypcji."""
        
        # Pobierz nowe fragmenty
        new_chunks = self._state.transcript_chunks[self._state.last_analysis_index:]
        if not new_chunks:
            return {"status": "no_new_content"}
        
        new_text = " ".join(new_chunks)
        self._state.last_analysis_index = len(self._state.transcript_chunks)
        
        # Przygotuj kontekst
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
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parsuj odpowiedź
        analysis = self._parse_analysis_response(response.content[0].text)
        
        # Aktualizuj stan na podstawie analizy
        self._apply_analysis(analysis)
        
        if self.on_update:
            self.on_update(self._state)
        
        return analysis
    
    def _parse_analysis_response(self, content: str) -> Dict[str, Any]:
        """Parsuj odpowiedź analizy."""
        import json
        try:
            # Szukaj JSON w odpowiedzi
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
        """Zastosuj wyniki analizy do stanu."""
        
        # Aktualizuj status tematów
        for topic_info in analysis.get("topics_discussed", []):
            topic_name = topic_info.get("topic", "")
            key_points = topic_info.get("key_points", [])
            
            for todo in self._state.todos:
                if self._topics_match(todo.topic, topic_name):
                    todo.mark_discussed(key_points)
                    break
        
        # Dodaj nowe sugestie
        for suggestion_data in analysis.get("suggestions", []):
            suggestion = Suggestion(
                type=suggestion_data.get("type", "question"),
                content=suggestion_data.get("content", ""),
                priority=suggestion_data.get("priority", "normal"),
                related_todo_id=suggestion_data.get("related_todo")
            )
            self._state.suggestions.append(suggestion)
        
        # Dodaj ostrzeżenia jako sugestie z wysokim priorytetem
        for warning in analysis.get("warnings", []):
            suggestion = Suggestion(
                type="warning",
                content=warning,
                priority="high"
            )
            self._state.suggestions.append(suggestion)
    
    def _topics_match(self, topic1: str, topic2: str) -> bool:
        """Sprawdź czy dwa tematy są podobne."""
        # Proste porównanie - możesz rozbudować o embeddingi
        t1 = topic1.lower().strip()
        t2 = topic2.lower().strip()
        
        # Dokładne dopasowanie
        if t1 == t2:
            return True
        
        # Częściowe dopasowanie
        if t1 in t2 or t2 in t1:
            return True
        
        # Sprawdź wspólne słowa kluczowe
        words1 = set(t1.split())
        words2 = set(t2.split())
        common = words1 & words2
        
        return len(common) >= 2
    
    async def _generate_final_summary(self) -> Dict[str, Any]:
        """Wygeneruj końcowe podsumowanie spotkania."""
        
        full_transcript = " ".join(self._state.transcript_chunks)
        
        prompt = self.prompts.final_summary_prompt(
            transcript=full_transcript,
            todos=self._state.todos,
            goals=self._state.goals
        )
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            "summary": response.content[0].text,
            "todos_completed": [
                t for t in self._state.todos 
                if t.status == TodoStatus.DISCUSSED
            ],
            "todos_missed": [
                t for t in self._state.todos 
                if t.status == TodoStatus.PENDING
            ],
            "total_suggestions": len(self._state.suggestions),
            "duration_minutes": (
                datetime.now() - self._state.started_at
            ).total_seconds() / 60
        }
```

### 4.3 Implementacja: `prompts.py`

```python
# backend/app/agent/prompts.py
"""
Prompty dla Meeting Agent.
"""
from typing import List, Optional


class AgentPrompts:
    """Zarządzanie promptami dla Meeting Agent."""
    
    def generate_todos_prompt(
        self,
        agenda: str,
        goals: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> str:
        """Prompt do generowania TODO listy z agendy."""
        
        goals_text = ""
        if goals:
            goals_text = f"""
## Cele do osiągnięcia:
{chr(10).join(f"- {g}" for g in goals)}
"""
        
        context_text = ""
        if context:
            context_text = f"""
## Kontekst (poprzednie spotkania, tło):
{context}
"""
        
        return f"""Jesteś asystentem spotkań. Na podstawie agendy stwórz listę konkretnych tematów/pytań do omówienia.

## Agenda spotkania:
{agenda}
{goals_text}
{context_text}

## Zadanie:
Przeanalizuj agendę i wygeneruj listę TODO - konkretnych punktów do omówienia podczas spotkania.

Dla każdego punktu określ:
- topic: Krótka nazwa tematu (max 10 słów)
- description: Co dokładnie należy omówić/ustalić
- priority: Priorytet 1-5 (1 = najważniejszy)

## Format odpowiedzi (JSON):
```json
[
  {{
    "topic": "Nazwa tematu",
    "description": "Co trzeba omówić, jakie pytania zadać, co ustalić",
    "priority": 1
  }},
  ...
]
```

Wygeneruj 5-15 konkretnych punktów TODO. Bądź praktyczny i konkretny.
Odpowiedz TYLKO JSONem, bez dodatkowego tekstu."""
    
    def analyze_transcript_prompt(
        self,
        transcript: str,
        pending_topics: List[str],
        discussed_topics: List[str],
        goals: List[str]
    ) -> str:
        """Prompt do analizy fragmentu transkrypcji."""
        
        return f"""Jesteś asystentem analizującym przebieg spotkania w czasie rzeczywistym.

## Nowy fragment transkrypcji:
{transcript}

## Tematy jeszcze NIE omówione:
{chr(10).join(f"- {t}" for t in pending_topics) if pending_topics else "- Wszystkie tematy omówione"}

## Tematy już omówione:
{chr(10).join(f"- {t}" for t in discussed_topics) if discussed_topics else "- Brak"}

## Cele spotkania:
{chr(10).join(f"- {g}" for g in goals) if goals else "- Nie określono"}

## Zadanie:
Przeanalizuj fragment i określ:
1. Które z nieomówionych tematów zostały poruszone w tym fragmencie
2. Jakie pytania warto zadać (follow-up do dyskusji)
3. Czy są jakieś ostrzeżenia (np. zbaczanie z tematu, pomijanie ważnych kwestii)

## Format odpowiedzi (JSON):
```json
{{
  "topics_discussed": [
    {{
      "topic": "Nazwa tematu z listy pending",
      "key_points": ["Główny punkt 1", "Główny punkt 2"],
      "completion": "full|partial"
    }}
  ],
  "suggestions": [
    {{
      "type": "question|followup|reminder",
      "content": "Treść sugestii/pytania",
      "priority": "low|normal|high",
      "related_todo": "nazwa_tematu lub null"
    }}
  ],
  "warnings": [
    "Ostrzeżenie jeśli coś wymaga uwagi"
  ]
}}
```

Odpowiedz TYLKO JSONem. Bądź zwięzły i praktyczny."""
    
    def final_summary_prompt(
        self,
        transcript: str,
        todos: List,
        goals: List[str]
    ) -> str:
        """Prompt do końcowego podsumowania spotkania."""
        
        todos_status = []
        for todo in todos:
            status_emoji = "✅" if todo.status.value == "discussed" else "❌"
            todos_status.append(f"{status_emoji} {todo.topic}")
        
        return f"""Jesteś asystentem tworzącym profesjonalne podsumowanie spotkania.

## Pełna transkrypcja spotkania:
{transcript[:15000]}  # Limit dla bezpieczeństwa

## Status tematów z agendy:
{chr(10).join(todos_status)}

## Cele spotkania:
{chr(10).join(f"- {g}" for g in goals) if goals else "- Nie określono"}

## Zadanie:
Stwórz profesjonalne podsumowanie spotkania zawierające:

1. **Streszczenie** (3-5 zdań)
2. **Kluczowe ustalenia** (bullet points)
3. **Action items** (kto, co, do kiedy)
4. **Otwarte kwestie** (co wymaga dalszej dyskusji)
5. **Następne kroki**

Pisz po polsku. Bądź konkretny i praktyczny. Używaj formatowania Markdown."""
    
    def suggest_questions_prompt(
        self,
        current_topic: str,
        transcript_excerpt: str,
        goals: List[str]
    ) -> str:
        """Prompt do sugerowania pytań w trakcie dyskusji."""
        
        return f"""Kontekst: Trwa dyskusja na temat "{current_topic}".

Fragment rozmowy:
{transcript_excerpt}

Cele spotkania:
{chr(10).join(f"- {g}" for g in goals) if goals else "- Ogólna dyskusja"}

Zasugeruj 2-3 konkretne pytania, które warto zadać, aby:
- Pogłębić temat
- Upewnić się, że nic nie pominięto
- Ustalić konkretne następne kroki

Format: Lista pytań, każde w nowej linii."""
```

### 4.4 Nowe endpointy dla Agent API

```python
# Dodaj do backend/app/main.py

from app.agent.meeting_agent import MeetingAgent, AgentState, TodoItem
from pydantic import BaseModel
from typing import Optional, List

# Models
class StartMeetingRequest(BaseModel):
    meeting_id: str
    agenda: str
    goals: Optional[List[str]] = None
    context: Optional[str] = None

class TranscriptChunkRequest(BaseModel):
    meeting_id: str
    text: str

# Global agent storage (in production use Redis/DB)
_active_agents: Dict[str, MeetingAgent] = {}


@app.post("/api/agent/start")
async def start_meeting_agent(request: StartMeetingRequest):
    """Uruchom agenta dla nowego spotkania."""
    
    agent = MeetingAgent(
        model="claude-sonnet-4-20250514",
        analysis_interval=45
    )
    
    state = await agent.start_meeting(
        meeting_id=request.meeting_id,
        agenda=request.agenda,
        goals=request.goals,
        context=request.context
    )
    
    _active_agents[request.meeting_id] = agent
    
    return {
        "status": "started",
        "meeting_id": request.meeting_id,
        "todos": [
            {"id": t.id, "topic": t.topic, "description": t.description, "priority": t.priority}
            for t in state.todos
        ]
    }


@app.post("/api/agent/transcript")
async def add_transcript_to_agent(request: TranscriptChunkRequest):
    """Dodaj fragment transkrypcji do analizy."""
    
    agent = _active_agents.get(request.meeting_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Meeting agent not found")
    
    await agent.add_transcript_chunk(request.text)
    return {"status": "received"}


@app.post("/api/agent/analyze/{meeting_id}")
async def force_agent_analysis(meeting_id: str):
    """Wymuś natychmiastową analizę."""
    
    agent = _active_agents.get(meeting_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Meeting agent not found")
    
    analysis = await agent.analyze_now()
    state = agent.get_state()
    
    return {
        "analysis": analysis,
        "todos": [
            {
                "id": t.id, 
                "topic": t.topic, 
                "status": t.status.value,
                "key_points": t.key_points
            }
            for t in state.todos
        ],
        "suggestions": [
            {"type": s.type, "content": s.content, "priority": s.priority}
            for s in agent.get_recent_suggestions(5)
        ]
    }


@app.get("/api/agent/state/{meeting_id}")
async def get_agent_state(meeting_id: str):
    """Pobierz aktualny stan agenta."""
    
    agent = _active_agents.get(meeting_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Meeting agent not found")
    
    state = agent.get_state()
    
    return {
        "meeting_id": state.meeting_id,
        "is_active": state.is_active,
        "todos": [
            {
                "id": t.id,
                "topic": t.topic,
                "description": t.description,
                "status": t.status.value,
                "priority": t.priority,
                "key_points": t.key_points
            }
            for t in state.todos
        ],
        "pending_count": len(agent.get_pending_todos()),
        "suggestions": [
            {"type": s.type, "content": s.content, "priority": s.priority}
            for s in agent.get_recent_suggestions(10)
        ]
    }


@app.post("/api/agent/end/{meeting_id}")
async def end_meeting_agent(meeting_id: str):
    """Zakończ spotkanie i pobierz podsumowanie."""
    
    agent = _active_agents.get(meeting_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Meeting agent not found")
    
    summary = await agent.end_meeting()
    
    # Cleanup
    del _active_agents[meeting_id]
    
    return summary
```

---

## 5. Faza 4: Aplikacja mobilna iOS

### 5.1 Szybka opcja: WhisperBoard fork

```bash
# Sklonuj WhisperBoard
git clone https://github.com/Saik0s/Whisperboard.git
cd Whisperboard

# Otwórz w Xcode
open WhisperBoard.xcodeproj
```

**Modyfikacje do zrobienia:**

1. **Dodaj networking do twojego backendu:**

```swift
// NetworkManager.swift
import Foundation

class MeetilyAPI {
    static let shared = MeetilyAPI()
    private let baseURL = "http://YOUR_MAC_IP:5167"
    
    func uploadAndTranscribe(audioURL: URL) async throws -> TranscriptionResult {
        let url = URL(string: "\(baseURL)/api/elevenlabs/transcribe")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        // Upload audio file
        let audioData = try Data(contentsOf: audioURL)
        // ... multipart form data ...
        
        let (data, _) = try await URLSession.shared.data(for: request)
        return try JSONDecoder().decode(TranscriptionResult.self, from: data)
    }
}
```

2. **Synchronizacja z Mac app** - przez iCloud lub REST API

### 5.2 Alternatywa: Prosta PWA

Jeśli nie chcesz native iOS app, możesz zrobić Progressive Web App:

```typescript
// frontend/src/app/mobile/page.tsx
"use client"

import { useState, useRef } from 'react'

export default function MobileRecorder() {
  const [isRecording, setIsRecording] = useState(false)
  const mediaRecorder = useRef<MediaRecorder | null>(null)
  const chunks = useRef<Blob[]>([])
  
  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    mediaRecorder.current = new MediaRecorder(stream)
    
    mediaRecorder.current.ondataavailable = (e) => {
      chunks.current.push(e.data)
    }
    
    mediaRecorder.current.onstop = async () => {
      const blob = new Blob(chunks.current, { type: 'audio/webm' })
      await uploadToBackend(blob)
    }
    
    mediaRecorder.current.start()
    setIsRecording(true)
  }
  
  const stopRecording = () => {
    mediaRecorder.current?.stop()
    setIsRecording(false)
  }
  
  const uploadToBackend = async (blob: Blob) => {
    const formData = new FormData()
    formData.append('audio', blob, 'recording.webm')
    
    await fetch('/api/elevenlabs/transcribe', {
      method: 'POST',
      body: formData
    })
  }
  
  return (
    <div className="mobile-recorder">
      <button onClick={isRecording ? stopRecording : startRecording}>
        {isRecording ? '⏹️ Stop' : '🎙️ Nagrywaj'}
      </button>
    </div>
  )
}
```

---

## 6. Szczegóły implementacji

### 6.1 Konfiguracja środowiska

```bash
# .env (root projektu)
# ElevenLabs
ELEVENLABS_API_KEY=your_api_key_here
ELEVENLABS_MODEL=scribe_v2
ELEVENLABS_DEFAULT_LANGUAGE=  # puste = auto-detect
ELEVENLABS_PRICING_TIER=pro

# Anthropic (dla Agent)
ANTHROPIC_API_KEY=your_anthropic_key_here

# App settings
MEETILY_DATA_DIR=~/.meetily
MEETILY_LOG_LEVEL=INFO
```

### 6.2 Modyfikacja UI - panel kosztów

```typescript
// frontend/src/components/CostPanel.tsx
import { useEffect, useState } from 'react'

interface CostStats {
  total_cost_usd: number
  total_duration_formatted: string
  current_month_cost: number
  current_month_hours: number
}

export function CostPanel() {
  const [stats, setStats] = useState<CostStats | null>(null)
  
  useEffect(() => {
    fetch('/api/elevenlabs/costs')
      .then(r => r.json())
      .then(setStats)
  }, [])
  
  if (!stats) return null
  
  return (
    <div className="cost-panel bg-gray-800 rounded-lg p-4">
      <h3 className="text-lg font-semibold mb-2">💰 Koszty ElevenLabs</h3>
      
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-gray-400 text-sm">Ta sesja</p>
          <p className="text-2xl font-bold">${stats.total_cost_usd.toFixed(2)}</p>
          <p className="text-sm text-gray-500">{stats.total_duration_formatted}</p>
        </div>
        
        <div>
          <p className="text-gray-400 text-sm">Ten miesiąc</p>
          <p className="text-2xl font-bold">${stats.current_month_cost.toFixed(2)}</p>
          <p className="text-sm text-gray-500">{stats.current_month_hours.toFixed(1)}h</p>
        </div>
      </div>
    </div>
  )
}
```

### 6.3 Modyfikacja UI - Agent panel

```typescript
// frontend/src/components/AgentPanel.tsx
import { useState, useEffect } from 'react'

interface Todo {
  id: string
  topic: string
  description: string
  status: 'pending' | 'in_progress' | 'discussed' | 'skipped'
  priority: number
  key_points: string[]
}

interface Suggestion {
  type: string
  content: string
  priority: string
}

export function AgentPanel({ meetingId }: { meetingId: string }) {
  const [todos, setTodos] = useState<Todo[]>([])
  const [suggestions, setSuggestions] = useState<Suggestion[]>([])
  const [isActive, setIsActive] = useState(false)
  
  // Polling for updates
  useEffect(() => {
    if (!isActive) return
    
    const interval = setInterval(async () => {
      const res = await fetch(`/api/agent/state/${meetingId}`)
      const data = await res.json()
      setTodos(data.todos)
      setSuggestions(data.suggestions)
    }, 5000)
    
    return () => clearInterval(interval)
  }, [meetingId, isActive])
  
  const startAgent = async (agenda: string, goals: string[]) => {
    await fetch('/api/agent/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ meeting_id: meetingId, agenda, goals })
    })
    setIsActive(true)
  }
  
  return (
    <div className="agent-panel">
      {/* Setup form when not active */}
      {!isActive && <AgentSetupForm onStart={startAgent} />}
      
      {/* Active meeting view */}
      {isActive && (
        <>
          {/* TODO List */}
          <div className="todos-section">
            <h3>📋 Do omówienia</h3>
            {todos.filter(t => t.status === 'pending').map(todo => (
              <TodoItem key={todo.id} todo={todo} />
            ))}
          </div>
          
          {/* Completed */}
          <div className="completed-section">
            <h3>✅ Omówione</h3>
            {todos.filter(t => t.status === 'discussed').map(todo => (
              <TodoItem key={todo.id} todo={todo} />
            ))}
          </div>
          
          {/* Suggestions */}
          <div className="suggestions-section">
            <h3>💡 Sugestie</h3>
            {suggestions.map((s, i) => (
              <SuggestionCard key={i} suggestion={s} />
            ))}
          </div>
        </>
      )}
    </div>
  )
}
```

---

## 7. Koszty i cennik

### Miesięczne koszty szacunkowe

| Komponent | Użycie | Koszt |
|-----------|--------|-------|
| **ElevenLabs Pro** | 300h/mies | $99/mies |
| **Anthropic Claude** | ~100k tokens/mies | ~$3/mies |
| **Hosting (opcjonalnie)** | VPS dla bota | $0-20/mies |
| **TOTAL** | | **~$102-122/mies** |

### Kalkulacja per spotkanie

```
1h spotkanie:
- ElevenLabs: 1h × $0.33 = $0.33
- Claude Agent: ~10 wywołań × ~2k tokens = ~$0.06
- TOTAL: ~$0.40 per godzinę spotkania
```

### Porównanie z alternatywami

| Narzędzie | Cena/mies | Jakość PL | Diaryzacja | Offline |
|-----------|-----------|-----------|------------|---------|
| **Twoje DIY** | ~$100 | ⭐⭐⭐⭐⭐ | ✅ | ✅ |
| Otter.ai Business | $20 | ❌ | ✅ | ✅ |
| Fireflies Business | $19 | ⭐⭐⭐ | ✅ | ✅ |
| MeetGeek Pro | $15 | ⭐⭐⭐ | ✅ | ✅ |
| Krisp Pro | $8 | ⭐⭐⭐⭐ | ✅ | ✅ |

---

## 🚀 Quick Start Checklist

```
□ 1. Fork repo: github.com/Zackriya-Solutions/meeting-minutes
□ 2. Clone i setup środowiska
□ 3. Utwórz .env z API keys
□ 4. Zaimplementuj backend/app/elevenlabs/
□ 5. Dodaj endpointy do main.py
□ 6. Zaimplementuj backend/app/agent/
□ 7. Dodaj komponenty UI (CostPanel, AgentPanel)
□ 8. Test end-to-end
□ 9. (Opcjonalnie) iOS app
```

---

## 📚 Referencje

- [ElevenLabs Scribe v2 API](https://elevenlabs.io/docs/api-reference/speech-to-text/convert)
- [Anthropic Claude API](https://docs.anthropic.com/en/api/getting-started)
- [Meetily GitHub](https://github.com/Zackriya-Solutions/meeting-minutes)
- [Tauri Documentation](https://tauri.app/v1/guides/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)