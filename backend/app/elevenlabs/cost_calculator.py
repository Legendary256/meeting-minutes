"""
ElevenLabs API Cost Calculator
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, date
import json
import os


@dataclass
class TranscriptionCostEntry:
    """Single cost entry."""
    timestamp: datetime
    duration_seconds: float
    cost_usd: float
    meeting_id: Optional[str] = None
    meeting_name: Optional[str] = None


@dataclass
class CostSummary:
    """Cost summary."""
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
    Calculator and tracker for ElevenLabs transcription costs.
    """
    
    def __init__(
        self, 
        price_per_hour: float = 0.33,
        history_file: Optional[str] = None
    ):
        """
        Args:
            price_per_hour: Price per hour of audio in USD
            history_file: Path to cost history file (JSON)
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
        Calculate cost for given audio duration.
        
        Args:
            duration_seconds: Audio duration in seconds
            
        Returns:
            Cost in USD
        """
        return duration_seconds * self.price_per_second
    
    def log_transcription(
        self,
        duration_seconds: float,
        cost_usd: float,
        meeting_id: Optional[str] = None,
        meeting_name: Optional[str] = None
    ):
        """Log transcription to history."""
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
        Get cost summary for a period.
        
        Args:
            period_start: Period start (None = from beginning)
            period_end: Period end (None = until now)
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
        """Get summary for a specific month."""
        from calendar import monthrange
        
        start = date(year, month, 1)
        _, last_day = monthrange(year, month)
        end = date(year, month, last_day)
        
        return self.get_summary(start, end)
    
    def get_current_month_summary(self) -> CostSummary:
        """Get summary for current month."""
        today = date.today()
        return self.get_monthly_summary(today.year, today.month)
    
    def estimate_monthly_cost(self, hours_per_month: float) -> float:
        """Estimate monthly cost for given number of hours."""
        return hours_per_month * self.price_per_hour
    
    def get_plan_recommendation(self, monthly_hours: float) -> Dict:
        """
        Recommend ElevenLabs plan based on monthly usage.
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
            # Exceeds Scale
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
        """Filter history by period."""
        entries = self._history
        
        if start:
            entries = [e for e in entries if e.timestamp.date() >= start]
        if end:
            entries = [e for e in entries if e.timestamp.date() <= end]
        
        return entries
    
    def _load_history(self):
        """Load history from file."""
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
        """Save history to file."""
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
