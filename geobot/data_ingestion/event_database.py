"""
Event Database for Geopolitical Intelligence

Persistent storage and querying for structured events.

Features:
- Efficient time-range queries
- Actor-based filtering
- Event type filtering
- Temporal aggregation
- Causal graph construction from events
- Export to panel data formats
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import pandas as pd

from .event_extraction import GeopoliticalEvent, EventType, TemporalNormalizer


class EventDatabase:
    """
    SQLite-based event database with efficient querying.
    """

    def __init__(self, db_path: str = "events.db"):
        """
        Initialize event database.

        Parameters
        ----------
        db_path : str
            Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Connect to database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def _create_tables(self):
        """Create database schema."""
        cursor = self.conn.cursor()

        # Events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                location TEXT,
                magnitude REAL,
                confidence REAL,
                source TEXT,
                text TEXT,
                metadata TEXT
            )
        ''')

        # Actors table (many-to-many with events)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS event_actors (
                event_id TEXT,
                actor TEXT,
                role TEXT,
                FOREIGN KEY (event_id) REFERENCES events(event_id),
                PRIMARY KEY (event_id, actor)
            )
        ''')

        # Causal relationships
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS causal_links (
                cause_event_id TEXT,
                effect_event_id TEXT,
                strength REAL,
                confidence REAL,
                FOREIGN KEY (cause_event_id) REFERENCES events(event_id),
                FOREIGN KEY (effect_event_id) REFERENCES events(event_id),
                PRIMARY KEY (cause_event_id, effect_event_id)
            )
        ''')

        # Create indices for fast queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_actor ON event_actors(actor)')

        self.conn.commit()

    def insert_event(self, event: GeopoliticalEvent) -> None:
        """
        Insert event into database.

        Parameters
        ----------
        event : GeopoliticalEvent
            Event to insert
        """
        cursor = self.conn.cursor()

        # Normalize timestamp
        timestamp_str = TemporalNormalizer.normalize_to_utc(event.timestamp).isoformat()

        # Insert main event
        cursor.execute('''
            INSERT OR REPLACE INTO events
            (event_id, timestamp, event_type, location, magnitude, confidence, source, text, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.event_id,
            timestamp_str,
            event.event_type.value,
            event.location,
            event.magnitude,
            event.confidence,
            event.source,
            event.text,
            json.dumps(event.metadata)
        ))

        # Insert actors
        for actor in event.actors:
            cursor.execute('''
                INSERT OR REPLACE INTO event_actors (event_id, actor, role)
                VALUES (?, ?, ?)
            ''', (event.event_id, actor, 'participant'))

        # Insert target as actor with different role
        if event.target:
            cursor.execute('''
                INSERT OR REPLACE INTO event_actors (event_id, actor, role)
                VALUES (?, ?, ?)
            ''', (event.event_id, event.target, 'target'))

        self.conn.commit()

    def insert_events(self, events: List[GeopoliticalEvent]) -> None:
        """
        Bulk insert events.

        Parameters
        ----------
        events : list
            List of events to insert
        """
        for event in events:
            self.insert_event(event)

    def query_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[EventType]] = None,
        actors: Optional[List[str]] = None,
        min_magnitude: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[GeopoliticalEvent]:
        """
        Query events with filters.

        Parameters
        ----------
        start_time : datetime, optional
            Start of time range
        end_time : datetime, optional
            End of time range
        event_types : list, optional
            Filter by event types
        actors : list, optional
            Filter by actors
        min_magnitude : float, optional
            Minimum magnitude
        limit : int, optional
            Maximum number of results

        Returns
        -------
        list
            List of matching events
        """
        cursor = self.conn.cursor()

        query = "SELECT DISTINCT e.* FROM events e"
        conditions = []
        params = []

        # Join with actors if needed
        if actors:
            query += " JOIN event_actors ea ON e.event_id = ea.event_id"

        # Time range
        if start_time:
            conditions.append("e.timestamp >= ?")
            params.append(start_time.isoformat())
        if end_time:
            conditions.append("e.timestamp <= ?")
            params.append(end_time.isoformat())

        # Event types
        if event_types:
            placeholders = ','.join('?' * len(event_types))
            conditions.append(f"e.event_type IN ({placeholders})")
            params.extend([et.value for et in event_types])

        # Actors
        if actors:
            placeholders = ','.join('?' * len(actors))
            conditions.append(f"ea.actor IN ({placeholders})")
            params.extend(actors)

        # Magnitude
        if min_magnitude is not None:
            conditions.append("e.magnitude >= ?")
            params.append(min_magnitude)

        # Build query
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY e.timestamp DESC"

        if limit:
            query += f" LIMIT {limit}"

        # Execute
        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Convert to GeopoliticalEvent objects
        events = []
        for row in rows:
            # Get actors
            cursor.execute(
                "SELECT actor FROM event_actors WHERE event_id = ?",
                (row['event_id'],)
            )
            actors_rows = cursor.fetchall()
            event_actors = [r['actor'] for r in actors_rows]

            # Reconstruct event
            event = GeopoliticalEvent(
                event_id=row['event_id'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                event_type=EventType(row['event_type']),
                actors=event_actors,
                location=row['location'],
                magnitude=row['magnitude'],
                confidence=row['confidence'],
                source=row['source'],
                text=row['text'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            events.append(event)

        return events

    def get_actor_timeline(
        self,
        actor: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[GeopoliticalEvent]:
        """
        Get timeline of events for a specific actor.

        Parameters
        ----------
        actor : str
            Actor name
        start_time : datetime, optional
            Start time
        end_time : datetime, optional
            End time

        Returns
        -------
        list
            Events involving actor
        """
        return self.query_events(
            start_time=start_time,
            end_time=end_time,
            actors=[actor]
        )

    def get_event_counts_by_type(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, int]:
        """
        Get event counts by type.

        Parameters
        ----------
        start_time : datetime, optional
            Start time
        end_time : datetime, optional
            End time

        Returns
        -------
        dict
            Counts by event type
        """
        cursor = self.conn.cursor()

        query = "SELECT event_type, COUNT(*) as count FROM events"
        conditions = []
        params = []

        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time.isoformat())
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time.isoformat())

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " GROUP BY event_type"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return {row['event_type']: row['count'] for row in rows}

    def aggregate_by_time(
        self,
        granularity: str = 'day',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[EventType]] = None
    ) -> pd.DataFrame:
        """
        Aggregate events by time period.

        Parameters
        ----------
        granularity : str
            Time granularity ('day', 'week', 'month')
        start_time : datetime, optional
            Start time
        end_time : datetime, optional
            End time
        event_types : list, optional
            Filter by event types

        Returns
        -------
        pd.DataFrame
            Time series of event counts
        """
        events = self.query_events(
            start_time=start_time,
            end_time=end_time,
            event_types=event_types
        )

        if not events:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'timestamp': e.timestamp,
                'event_type': e.event_type.value,
                'magnitude': e.magnitude
            }
            for e in events
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        # Resample
        if granularity == 'day':
            freq = 'D'
        elif granularity == 'week':
            freq = 'W'
        elif granularity == 'month':
            freq = 'M'
        else:
            raise ValueError(f"Unknown granularity: {granularity}")

        # Aggregate
        aggregated = df.resample(freq).agg({
            'magnitude': ['count', 'mean', 'sum']
        })

        return aggregated

    def export_to_panel_data(
        self,
        actors: List[str],
        start_time: datetime,
        end_time: datetime,
        granularity: str = 'day'
    ) -> Dict[str, pd.DataFrame]:
        """
        Export to panel data format.

        Parameters
        ----------
        actors : list
            List of actors
        start_time : datetime
            Start time
        end_time : datetime
            End time
        granularity : str
            Time granularity

        Returns
        -------
        dict
            Panel data {actor: DataFrame}
        """
        from .event_extraction import CausalFeatureExtractor

        # Get events for each actor
        panel = {}
        for actor in actors:
            events = self.get_actor_timeline(actor, start_time, end_time)

            # Extract features
            extractor = CausalFeatureExtractor()
            panel_data = extractor.construct_panel_data([events], [actor], granularity)

            if actor in panel_data:
                panel[actor] = panel_data[actor]

        return panel

    def add_causal_link(
        self,
        cause_event_id: str,
        effect_event_id: str,
        strength: float = 1.0,
        confidence: float = 0.5
    ) -> None:
        """
        Add causal link between events.

        Parameters
        ----------
        cause_event_id : str
            ID of cause event
        effect_event_id : str
            ID of effect event
        strength : float
            Causal strength
        confidence : float
            Confidence in link
        """
        cursor = self.conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO causal_links
            (cause_event_id, effect_event_id, strength, confidence)
            VALUES (?, ?, ?, ?)
        ''', (cause_event_id, effect_event_id, strength, confidence))

        self.conn.commit()

    def get_causal_graph(self) -> Dict[str, List[str]]:
        """
        Get causal graph from event links.

        Returns
        -------
        dict
            Adjacency list representation
        """
        cursor = self.conn.cursor()

        cursor.execute("SELECT cause_event_id, effect_event_id FROM causal_links")
        rows = cursor.fetchall()

        graph = {}
        for row in rows:
            cause = row['cause_event_id']
            effect = row['effect_event_id']

            if cause not in graph:
                graph[cause] = []
            graph[cause].append(effect)

        return graph

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class EventStream:
    """
    Real-time event stream processor.

    Monitors and processes incoming events in real-time.
    """

    def __init__(self, db: EventDatabase):
        """
        Initialize event stream.

        Parameters
        ----------
        db : EventDatabase
            Event database
        """
        self.db = db
        self.subscribers = []

    def subscribe(self, callback: callable) -> None:
        """
        Subscribe to event stream.

        Parameters
        ----------
        callback : callable
            Function to call on new events
        """
        self.subscribers.append(callback)

    def process_event(self, event: GeopoliticalEvent) -> None:
        """
        Process and store new event.

        Parameters
        ----------
        event : GeopoliticalEvent
            New event
        """
        # Store in database
        self.db.insert_event(event)

        # Notify subscribers
        for callback in self.subscribers:
            callback(event)

    def process_batch(self, events: List[GeopoliticalEvent]) -> None:
        """
        Process batch of events.

        Parameters
        ----------
        events : list
            List of events
        """
        self.db.insert_events(events)

        for event in events:
            for callback in self.subscribers:
                callback(event)
