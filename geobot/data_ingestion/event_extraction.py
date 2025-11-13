"""
Structured Event Extraction Pipeline

Converts unstructured intelligence (PDFs, articles, reports) into structured,
timestamped events suitable for:
- Causal graph construction and updates
- Time-series analysis
- Panel data modeling
- Temporal feature engineering

Event schema:
- Timestamp (normalized to UTC)
- Event type (conflict, diplomacy, economic, etc.)
- Actors (countries, organizations)
- Location (geospatial)
- Magnitude/severity
- Source and confidence
- Causal attributes (preconditions, effects)
"""

import re
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json


class EventType(Enum):
    """Event taxonomy."""
    CONFLICT = "conflict"
    DIPLOMACY = "diplomacy"
    ECONOMIC = "economic"
    MILITARY_MOBILIZATION = "military_mobilization"
    SANCTIONS = "sanctions"
    ALLIANCE_FORMATION = "alliance_formation"
    TREATY_SIGNING = "treaty_signing"
    PROTEST = "protest"
    ELECTION = "election"
    COUP = "coup"
    TERROR_ATTACK = "terror_attack"
    CYBER_ATTACK = "cyber_attack"
    TRADE_AGREEMENT = "trade_agreement"
    ARMS_DEAL = "arms_deal"
    HUMANITARIAN_CRISIS = "humanitarian_crisis"
    OTHER = "other"


@dataclass
class GeopoliticalEvent:
    """
    Structured geopolitical event.

    Attributes
    ----------
    event_id : str
        Unique event identifier
    timestamp : datetime
        Event timestamp (normalized to UTC)
    event_type : EventType
        Type of event
    actors : List[str]
        Involved actors (countries, organizations)
    target : Optional[str]
        Target of action (if applicable)
    location : Optional[str]
        Geographic location
    magnitude : float
        Event magnitude/severity (0-1)
    confidence : float
        Extraction confidence (0-1)
    source : str
        Source document/article
    text : str
        Original text describing event
    causal_preconditions : List[str]
        Identified preconditions
    causal_effects : List[str]
        Identified effects
    metadata : Dict[str, Any]
        Additional metadata
    """
    event_id: str
    timestamp: datetime
    event_type: EventType
    actors: List[str]
    target: Optional[str] = None
    location: Optional[str] = None
    magnitude: float = 0.5
    confidence: float = 0.5
    source: str = ""
    text: str = ""
    causal_preconditions: List[str] = field(default_factory=list)
    causal_effects: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'actors': self.actors,
            'target': self.target,
            'location': self.location,
            'magnitude': self.magnitude,
            'confidence': self.confidence,
            'source': self.source,
            'text': self.text,
            'causal_preconditions': self.causal_preconditions,
            'causal_effects': self.causal_effects,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeopoliticalEvent':
        """Load from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['event_type'] = EventType(data['event_type'])
        return cls(**data)


class EventExtractor:
    """
    Extract structured events from unstructured text.

    Uses rule-based patterns and NLP to identify:
    - Event mentions
    - Temporal expressions
    - Actor identification
    - Event type classification
    """

    def __init__(self):
        """Initialize event extractor."""
        self.country_names = self._load_country_names()
        self.organization_names = self._load_organization_names()
        self.event_patterns = self._compile_event_patterns()

    def _load_country_names(self) -> List[str]:
        """Load list of country names."""
        # Extended list of countries
        return [
            'United States', 'USA', 'China', 'Russia', 'India', 'Pakistan',
            'Iran', 'North Korea', 'South Korea', 'Japan', 'Germany', 'France',
            'United Kingdom', 'UK', 'Israel', 'Saudi Arabia', 'Turkey', 'Egypt',
            'Syria', 'Iraq', 'Afghanistan', 'Ukraine', 'Poland', 'Italy', 'Spain',
            'Canada', 'Australia', 'Brazil', 'Mexico', 'South Africa', 'Nigeria'
        ]

    def _load_organization_names(self) -> List[str]:
        """Load list of international organizations."""
        return [
            'NATO', 'UN', 'United Nations', 'EU', 'European Union',
            'OPEC', 'ASEAN', 'African Union', 'Arab League', 'G7', 'G20',
            'IMF', 'World Bank', 'WTO', 'WHO', 'ICC'
        ]

    def _compile_event_patterns(self) -> Dict[EventType, List[re.Pattern]]:
        """Compile regex patterns for event types."""
        patterns = {
            EventType.CONFLICT: [
                re.compile(r'\b(attack|strike|bomb|missile|war|combat|clash|battle)\b', re.I),
                re.compile(r'\b(invasion|offensive|assault|raid)\b', re.I)
            ],
            EventType.DIPLOMACY: [
                re.compile(r'\b(negotiation|talk|summit|meeting|dialogue)\b', re.I),
                re.compile(r'\b(diplomatic|embassy|ambassador)\b', re.I)
            ],
            EventType.SANCTIONS: [
                re.compile(r'\b(sanction|embargo|restriction|ban)\b', re.I)
            ],
            EventType.MILITARY_MOBILIZATION: [
                re.compile(r'\b(mobiliz|deploy|troop|force|military)\b', re.I)
            ],
            EventType.ALLIANCE_FORMATION: [
                re.compile(r'\b(alliance|partnership|coalition|pact)\b', re.I)
            ],
            EventType.TREATY_SIGNING: [
                re.compile(r'\b(treaty|agreement|accord|convention)\b', re.I)
            ],
            EventType.ELECTION: [
                re.compile(r'\b(election|vote|ballot|referendum)\b', re.I)
            ],
            EventType.COUP: [
                re.compile(r'\b(coup|overthrow|takeover|regime change)\b', re.I)
            ],
            EventType.TERROR_ATTACK: [
                re.compile(r'\b(terror|terrorist|extremist|bombing)\b', re.I)
            ],
            EventType.CYBER_ATTACK: [
                re.compile(r'\b(cyber|hack|breach|malware|ransomware)\b', re.I)
            ]
        }
        return patterns

    def extract_events(
        self,
        text: str,
        source: str = "",
        default_timestamp: Optional[datetime] = None
    ) -> List[GeopoliticalEvent]:
        """
        Extract events from text.

        Parameters
        ----------
        text : str
            Input text
        source : str
            Source identifier
        default_timestamp : datetime, optional
            Default timestamp if none found

        Returns
        -------
        list
            List of extracted events
        """
        events = []

        # Split into sentences
        sentences = self._split_sentences(text)

        for i, sentence in enumerate(sentences):
            # Detect event type
            event_type = self._classify_event_type(sentence)

            if event_type != EventType.OTHER:
                # Extract actors
                actors = self._extract_actors(sentence)

                # Extract timestamp
                timestamp = self._extract_timestamp(sentence, default_timestamp)

                # Extract location
                location = self._extract_location(sentence)

                # Compute magnitude
                magnitude = self._estimate_magnitude(sentence, event_type)

                # Create event
                event = GeopoliticalEvent(
                    event_id=f"{source}_{i}",
                    timestamp=timestamp,
                    event_type=event_type,
                    actors=actors,
                    location=location,
                    magnitude=magnitude,
                    confidence=0.7,  # Rule-based confidence
                    source=source,
                    text=sentence
                )

                events.append(event)

        return events

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def _classify_event_type(self, text: str) -> EventType:
        """Classify event type using patterns."""
        for event_type, patterns in self.event_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    return event_type
        return EventType.OTHER

    def _extract_actors(self, text: str) -> List[str]:
        """Extract actor entities."""
        actors = []

        # Check for countries
        for country in self.country_names:
            if country.lower() in text.lower():
                actors.append(country)

        # Check for organizations
        for org in self.organization_names:
            if org.lower() in text.lower():
                actors.append(org)

        return list(set(actors))  # Remove duplicates

    def _extract_timestamp(
        self,
        text: str,
        default: Optional[datetime] = None
    ) -> datetime:
        """Extract timestamp from text."""
        # Try to find date patterns
        date_patterns = [
            r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
            r'(\d{1,2})/(\d{1,2})/(\d{4})',  # MM/DD/YYYY
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})'
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                # Parse date (simplified)
                try:
                    date_str = match.group(0)
                    # Try multiple formats
                    for fmt in ['%Y-%m-%d', '%m/%d/%Y']:
                        try:
                            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
                        except:
                            continue
                except:
                    pass

        # Default to current time or provided default
        return default or datetime.now(timezone.utc)

    def _extract_location(self, text: str) -> Optional[str]:
        """Extract location from text."""
        # Check for country names as locations
        for country in self.country_names:
            if country.lower() in text.lower():
                return country

        return None

    def _estimate_magnitude(self, text: str, event_type: EventType) -> float:
        """Estimate event magnitude/severity."""
        # Keywords indicating severity
        high_severity_words = ['major', 'massive', 'large-scale', 'significant', 'devastating']
        low_severity_words = ['minor', 'small', 'limited', 'isolated']

        text_lower = text.lower()

        if any(word in text_lower for word in high_severity_words):
            return 0.8
        elif any(word in text_lower for word in low_severity_words):
            return 0.3
        else:
            return 0.5  # Default


class TemporalNormalizer:
    """
    Normalize timestamps to consistent format (UTC).

    Handles:
    - Time zone conversion
    - Temporal granularity (day, week, month)
    - Missing timestamps (imputation)
    """

    @staticmethod
    def normalize_to_utc(dt: datetime) -> datetime:
        """
        Normalize datetime to UTC.

        Parameters
        ----------
        dt : datetime
            Input datetime

        Returns
        -------
        datetime
            UTC datetime
        """
        if dt.tzinfo is None:
            # Assume local time
            return dt.replace(tzinfo=timezone.utc)
        else:
            return dt.astimezone(timezone.utc)

    @staticmethod
    def round_to_day(dt: datetime) -> datetime:
        """Round to start of day."""
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def round_to_week(dt: datetime) -> datetime:
        """Round to start of week (Monday)."""
        day_of_week = dt.weekday()
        days_to_subtract = day_of_week
        week_start = dt - datetime.timedelta(days=days_to_subtract)
        return TemporalNormalizer.round_to_day(week_start)


class CausalFeatureExtractor:
    """
    Extract causal features from events for modeling.

    Constructs features suitable for:
    - Causal graph learning
    - Structural equation modeling
    - Time-series forecasting
    """

    def __init__(self):
        """Initialize causal feature extractor."""
        pass

    def extract_features(
        self,
        events: List[GeopoliticalEvent],
        time_window: int = 30
    ) -> Dict[str, np.ndarray]:
        """
        Extract causal features from event sequence.

        Parameters
        ----------
        events : list
            List of events
        time_window : int
            Time window in days

        Returns
        -------
        dict
            Feature dictionary
        """
        import numpy as np

        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        # Count events by type
        event_counts = {}
        for event_type in EventType:
            event_counts[event_type.value] = sum(
                1 for e in sorted_events if e.event_type == event_type
            )

        # Actor involvement matrix
        all_actors = list(set(actor for e in sorted_events for actor in e.actors))
        actor_indices = {actor: i for i, actor in enumerate(all_actors)}

        # Event-actor matrix
        n_events = len(sorted_events)
        n_actors = len(all_actors)
        actor_matrix = np.zeros((n_events, n_actors))

        for i, event in enumerate(sorted_events):
            for actor in event.actors:
                if actor in actor_indices:
                    actor_matrix[i, actor_indices[actor]] = 1

        # Temporal features
        if sorted_events:
            time_deltas = []
            for i in range(1, len(sorted_events)):
                delta = (sorted_events[i].timestamp - sorted_events[i-1].timestamp).total_seconds() / 86400  # days
                time_deltas.append(delta)
            mean_time_delta = np.mean(time_deltas) if time_deltas else 0
        else:
            mean_time_delta = 0

        features = {
            'event_counts': np.array([event_counts[et.value] for et in EventType]),
            'actor_matrix': actor_matrix,
            'mean_time_delta': mean_time_delta,
            'total_events': n_events,
            'unique_actors': n_actors
        }

        return features

    def construct_panel_data(
        self,
        events: List[GeopoliticalEvent],
        actors: List[str],
        time_granularity: str = 'day'
    ) -> Dict[str, Any]:
        """
        Construct panel data structure from events.

        Panel data format: (actor, time) -> features

        Parameters
        ----------
        events : list
            List of events
        actors : list
            List of actors
        time_granularity : str
            Time granularity ('day', 'week', 'month')

        Returns
        -------
        dict
            Panel data structure
        """
        import pandas as pd
        import numpy as np

        # Create time index
        if not events:
            return {}

        sorted_events = sorted(events, key=lambda e: e.timestamp)
        start_time = sorted_events[0].timestamp
        end_time = sorted_events[-1].timestamp

        # Generate time grid
        if time_granularity == 'day':
            time_index = pd.date_range(start_time, end_time, freq='D')
        elif time_granularity == 'week':
            time_index = pd.date_range(start_time, end_time, freq='W')
        elif time_granularity == 'month':
            time_index = pd.date_range(start_time, end_time, freq='M')
        else:
            raise ValueError(f"Unknown granularity: {time_granularity}")

        # Initialize panel
        panel = {}
        for actor in actors:
            panel[actor] = pd.DataFrame(index=time_index, columns=['event_count', 'avg_magnitude'])
            panel[actor] = panel[actor].fillna(0)

        # Fill panel with events
        for event in sorted_events:
            for actor in event.actors:
                if actor in panel:
                    # Find closest time point
                    event_date = pd.Timestamp(event.timestamp)
                    closest_idx = time_index[np.argmin(np.abs(time_index - event_date))]

                    panel[actor].loc[closest_idx, 'event_count'] += 1
                    panel[actor].loc[closest_idx, 'avg_magnitude'] += event.magnitude

        # Normalize magnitudes
        for actor in actors:
            mask = panel[actor]['event_count'] > 0
            panel[actor].loc[mask, 'avg_magnitude'] /= panel[actor].loc[mask, 'event_count']

        return panel
