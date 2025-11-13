"""
Answer Quality Database for GeoBotv1

Logs all analyst responses with quality ratings for training data generation.

Features:
- Automatic logging of all Q&A interactions
- Multi-dimensional quality rating system
- Dataset export for supervised fine-tuning
- RLHF (Reinforcement Learning from Human Feedback) dataset preparation
- Good/bad answer separation for training
- Version tracking and A/B comparison

Use Cases:
- Build training dataset from production queries
- Identify common failure modes
- Create preference pairs for RLHF
- Fine-tune domain-specific models
- Benchmark model improvements over time
"""

import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import hashlib


@dataclass
class AnswerQuality:
    """Multi-dimensional quality assessment of an answer."""

    # Objective metrics (0-1 scale)
    factual_accuracy: float  # Are claims factually correct?
    analytical_rigor: float  # Is analysis sound and well-reasoned?
    source_reliability: float  # Are sources credible and cited?
    completeness: float  # Does it fully address the question?
    clarity: float  # Is it clear and well-structured?

    # Subjective ratings (0-1 scale)
    analyst_usefulness: float  # How useful to analyst?
    actionability: float  # Can user act on this?
    confidence_appropriate: float  # Is confidence level justified?

    # Overall score (computed or explicit)
    overall_score: float = 0.0

    # Feedback notes
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    feedback_notes: str = ""

    # Metadata
    rated_by: str = "auto"  # "auto", "analyst", "supervisor"
    rated_at: Optional[datetime] = None

    def __post_init__(self):
        if self.overall_score == 0.0:
            # Compute weighted average if not provided
            weights = {
                'factual_accuracy': 0.25,
                'analytical_rigor': 0.20,
                'source_reliability': 0.15,
                'completeness': 0.15,
                'clarity': 0.10,
                'analyst_usefulness': 0.10,
                'actionability': 0.05
            }

            self.overall_score = (
                self.factual_accuracy * weights['factual_accuracy'] +
                self.analytical_rigor * weights['analytical_rigor'] +
                self.source_reliability * weights['source_reliability'] +
                self.completeness * weights['completeness'] +
                self.clarity * weights['clarity'] +
                self.analyst_usefulness * weights['analyst_usefulness'] +
                self.actionability * weights['actionability']
            )

        if self.rated_at is None:
            self.rated_at = datetime.utcnow()

    def is_good(self, threshold: float = 0.7) -> bool:
        """Classify as 'good' answer based on threshold."""
        return self.overall_score >= threshold

    def is_bad(self, threshold: float = 0.4) -> bool:
        """Classify as 'bad' answer based on threshold."""
        return self.overall_score < threshold

    def quality_category(self) -> str:
        """Return quality category: excellent, good, acceptable, poor."""
        if self.overall_score >= 0.85:
            return "excellent"
        elif self.overall_score >= 0.70:
            return "good"
        elif self.overall_score >= 0.50:
            return "acceptable"
        else:
            return "poor"


@dataclass
class AnswerRecord:
    """Complete record of a Q&A interaction."""

    # Unique identifier
    record_id: str

    # Query information
    query: str
    query_intent: Dict[str, Any]
    query_timestamp: datetime

    # Answer information
    answer_narrative: str
    answer_structured: Dict[str, Any]
    answer_timestamp: datetime

    # Model/system information
    model_used: str  # "mistral-7b", "gpt-4", etc.
    modules_invoked: List[str]  # ["hawkes", "var", "bayesian"]
    processing_time_seconds: float

    # Quality assessment
    quality: Optional[AnswerQuality] = None

    # Context
    session_id: str = ""
    analyst_id: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Version tracking
    version: int = 1
    parent_record_id: Optional[str] = None  # For regenerated answers

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime to ISO format
        data['query_timestamp'] = self.query_timestamp.isoformat()
        data['answer_timestamp'] = self.answer_timestamp.isoformat()
        if self.quality and self.quality.rated_at:
            data['quality']['rated_at'] = self.quality.rated_at.isoformat()
        return data

    def to_training_example(self, format: str = "chat") -> Dict[str, Any]:
        """
        Convert to training data format.

        Args:
            format: "chat", "completion", "rlhf", "dpo"
        """
        if format == "chat":
            # ChatML format
            return {
                "messages": [
                    {"role": "system", "content": "You are a geopolitical intelligence analyst using mathematical forecasting methods."},
                    {"role": "user", "content": self.query},
                    {"role": "assistant", "content": self.answer_narrative}
                ],
                "quality_score": self.quality.overall_score if self.quality else None
            }

        elif format == "completion":
            # Completion format
            return {
                "prompt": f"Geopolitical Analysis Query: {self.query}\n\nAnalysis:",
                "completion": self.answer_narrative,
                "quality_score": self.quality.overall_score if self.quality else None
            }

        elif format == "rlhf":
            # RLHF preference format (needs comparison)
            return {
                "query": self.query,
                "response": self.answer_narrative,
                "score": self.quality.overall_score if self.quality else 0.5
            }

        elif format == "dpo":
            # Direct Preference Optimization format (needs pair)
            return {
                "prompt": self.query,
                "chosen": self.answer_narrative if self.quality and self.quality.is_good() else None,
                "rejected": self.answer_narrative if self.quality and self.quality.is_bad() else None,
                "score": self.quality.overall_score if self.quality else None
            }

        else:
            raise ValueError(f"Unknown format: {format}")


class AnswerDatabase:
    """
    Database for storing and analyzing Q&A interactions.

    Example:
        >>> db = AnswerDatabase("answers.db")
        >>>
        >>> # Log an answer from analyst agent
        >>> record = db.log_answer(
        ...     query="What is Iran nuclear risk?",
        ...     query_intent={'entities': ['Iran'], 'type': 'risk_assessment'},
        ...     answer_narrative="Based on analysis...",
        ...     answer_structured={'risk_score': 0.68},
        ...     model_used="mistral-7b",
        ...     modules_invoked=["hawkes", "var"]
        ... )
        >>>
        >>> # Rate the answer
        >>> quality = AnswerQuality(
        ...     factual_accuracy=0.85,
        ...     analytical_rigor=0.80,
        ...     source_reliability=0.75,
        ...     completeness=0.90,
        ...     clarity=0.88,
        ...     analyst_usefulness=0.92,
        ...     actionability=0.78,
        ...     confidence_appropriate=0.85
        ... )
        >>> db.rate_answer(record.record_id, quality)
        >>>
        >>> # Export good answers for training
        >>> good_dataset = db.export_training_dataset(
        ...     quality_threshold=0.7,
        ...     format="chat"
        ... )
    """

    def __init__(self, db_path: str = "answer_database.db"):
        """
        Initialize answer database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._create_schema()

    def _create_schema(self) -> None:
        """Create database schema."""
        cursor = self.conn.cursor()

        # Main answers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS answers (
                record_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                query_intent TEXT,
                query_timestamp TEXT NOT NULL,
                answer_narrative TEXT NOT NULL,
                answer_structured TEXT,
                answer_timestamp TEXT NOT NULL,
                model_used TEXT,
                modules_invoked TEXT,
                processing_time_seconds REAL,
                session_id TEXT,
                analyst_id TEXT,
                tags TEXT,
                metadata TEXT,
                version INTEGER DEFAULT 1,
                parent_record_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Quality ratings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_ratings (
                rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT NOT NULL,
                factual_accuracy REAL,
                analytical_rigor REAL,
                source_reliability REAL,
                completeness REAL,
                clarity REAL,
                analyst_usefulness REAL,
                actionability REAL,
                confidence_appropriate REAL,
                overall_score REAL,
                strengths TEXT,
                weaknesses TEXT,
                feedback_notes TEXT,
                rated_by TEXT,
                rated_at TEXT,
                FOREIGN KEY (record_id) REFERENCES answers(record_id)
            )
        ''')

        # Indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_query_timestamp ON answers(query_timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_used ON answers(model_used)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON answers(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_overall_score ON quality_ratings(overall_score)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_record_quality ON quality_ratings(record_id)')

        self.conn.commit()

    def log_answer(
        self,
        query: str,
        query_intent: Dict[str, Any],
        answer_narrative: str,
        answer_structured: Dict[str, Any],
        model_used: str,
        modules_invoked: List[str],
        processing_time_seconds: float = 0.0,
        session_id: str = "",
        analyst_id: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_rate: bool = True
    ) -> AnswerRecord:
        """
        Log a Q&A interaction to database.

        Args:
            query: User query
            query_intent: Parsed intent
            answer_narrative: Text answer
            answer_structured: Structured analysis data
            model_used: Model identifier
            modules_invoked: GeoBotv1 modules used
            processing_time_seconds: Processing time
            session_id: Session identifier
            analyst_id: Analyst identifier
            tags: Tags for categorization
            metadata: Additional metadata
            auto_rate: Automatically generate quality rating

        Returns:
            AnswerRecord with database ID
        """
        record_id = self._generate_record_id(query, answer_narrative)

        now = datetime.utcnow()

        record = AnswerRecord(
            record_id=record_id,
            query=query,
            query_intent=query_intent,
            query_timestamp=now,
            answer_narrative=answer_narrative,
            answer_structured=answer_structured,
            answer_timestamp=now,
            model_used=model_used,
            modules_invoked=modules_invoked,
            processing_time_seconds=processing_time_seconds,
            session_id=session_id,
            analyst_id=analyst_id,
            tags=tags or [],
            metadata=metadata or {}
        )

        # Insert into database
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO answers (
                record_id, query, query_intent, query_timestamp,
                answer_narrative, answer_structured, answer_timestamp,
                model_used, modules_invoked, processing_time_seconds,
                session_id, analyst_id, tags, metadata, version, parent_record_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record.record_id,
            record.query,
            json.dumps(record.query_intent),
            record.query_timestamp.isoformat(),
            record.answer_narrative,
            json.dumps(record.answer_structured),
            record.answer_timestamp.isoformat(),
            record.model_used,
            json.dumps(record.modules_invoked),
            record.processing_time_seconds,
            record.session_id,
            record.analyst_id,
            json.dumps(record.tags),
            json.dumps(record.metadata),
            record.version,
            record.parent_record_id
        ))
        self.conn.commit()

        # Auto-rate if requested
        if auto_rate:
            quality = self._auto_rate_answer(record)
            self.rate_answer(record_id, quality)
            record.quality = quality

        return record

    def rate_answer(self, record_id: str, quality: AnswerQuality) -> None:
        """
        Add or update quality rating for an answer.

        Args:
            record_id: Record identifier
            quality: Quality assessment
        """
        cursor = self.conn.cursor()

        # Check if rating exists
        cursor.execute('SELECT rating_id FROM quality_ratings WHERE record_id = ?', (record_id,))
        existing = cursor.fetchone()

        if existing:
            # Update existing rating
            cursor.execute('''
                UPDATE quality_ratings SET
                    factual_accuracy = ?,
                    analytical_rigor = ?,
                    source_reliability = ?,
                    completeness = ?,
                    clarity = ?,
                    analyst_usefulness = ?,
                    actionability = ?,
                    confidence_appropriate = ?,
                    overall_score = ?,
                    strengths = ?,
                    weaknesses = ?,
                    feedback_notes = ?,
                    rated_by = ?,
                    rated_at = ?
                WHERE record_id = ?
            ''', (
                quality.factual_accuracy,
                quality.analytical_rigor,
                quality.source_reliability,
                quality.completeness,
                quality.clarity,
                quality.analyst_usefulness,
                quality.actionability,
                quality.confidence_appropriate,
                quality.overall_score,
                json.dumps(quality.strengths),
                json.dumps(quality.weaknesses),
                quality.feedback_notes,
                quality.rated_by,
                quality.rated_at.isoformat(),
                record_id
            ))
        else:
            # Insert new rating
            cursor.execute('''
                INSERT INTO quality_ratings (
                    record_id, factual_accuracy, analytical_rigor,
                    source_reliability, completeness, clarity,
                    analyst_usefulness, actionability, confidence_appropriate,
                    overall_score, strengths, weaknesses, feedback_notes,
                    rated_by, rated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record_id,
                quality.factual_accuracy,
                quality.analytical_rigor,
                quality.source_reliability,
                quality.completeness,
                quality.clarity,
                quality.analyst_usefulness,
                quality.actionability,
                quality.confidence_appropriate,
                quality.overall_score,
                json.dumps(quality.strengths),
                json.dumps(quality.weaknesses),
                quality.feedback_notes,
                quality.rated_by,
                quality.rated_at.isoformat()
            ))

        self.conn.commit()

    def get_record(self, record_id: str) -> Optional[AnswerRecord]:
        """Retrieve a specific answer record."""
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT a.*, q.* FROM answers a
            LEFT JOIN quality_ratings q ON a.record_id = q.record_id
            WHERE a.record_id = ?
        ''', (record_id,))

        row = cursor.fetchone()
        if not row:
            return None

        return self._row_to_record(row)

    def get_good_answers(
        self,
        threshold: float = 0.7,
        limit: Optional[int] = None
    ) -> List[AnswerRecord]:
        """
        Get all answers rated above quality threshold.

        Args:
            threshold: Minimum overall_score
            limit: Maximum number of results

        Returns:
            List of high-quality answer records
        """
        cursor = self.conn.cursor()

        query = '''
            SELECT a.*, q.* FROM answers a
            INNER JOIN quality_ratings q ON a.record_id = q.record_id
            WHERE q.overall_score >= ?
            ORDER BY q.overall_score DESC
        '''

        if limit:
            query += f' LIMIT {limit}'

        cursor.execute(query, (threshold,))
        rows = cursor.fetchall()

        return [self._row_to_record(row) for row in rows]

    def get_bad_answers(
        self,
        threshold: float = 0.4,
        limit: Optional[int] = None
    ) -> List[AnswerRecord]:
        """
        Get all answers rated below quality threshold.

        Args:
            threshold: Maximum overall_score
            limit: Maximum number of results

        Returns:
            List of low-quality answer records
        """
        cursor = self.conn.cursor()

        query = '''
            SELECT a.*, q.* FROM answers a
            INNER JOIN quality_ratings q ON a.record_id = q.record_id
            WHERE q.overall_score < ?
            ORDER BY q.overall_score ASC
        '''

        if limit:
            query += f' LIMIT {limit}'

        cursor.execute(query, (threshold,))
        rows = cursor.fetchall()

        return [self._row_to_record(row) for row in rows]

    def export_training_dataset(
        self,
        quality_threshold: float = 0.7,
        format: str = "chat",
        output_path: Optional[str] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Export training dataset in specified format.

        Args:
            quality_threshold: Minimum quality score
            format: "chat", "completion", "rlhf", "dpo"
            output_path: Optional file path to save JSON
            include_metadata: Include quality scores and metadata

        Returns:
            List of training examples
        """
        good_answers = self.get_good_answers(threshold=quality_threshold)

        dataset = []
        for record in good_answers:
            example = record.to_training_example(format=format)

            if include_metadata:
                example['metadata'] = {
                    'record_id': record.record_id,
                    'quality_score': record.quality.overall_score if record.quality else None,
                    'modules_used': record.modules_invoked,
                    'timestamp': record.answer_timestamp.isoformat()
                }

            dataset.append(example)

        # Save to file if requested
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(dataset, f, indent=2)

        return dataset

    def export_preference_pairs(
        self,
        good_threshold: float = 0.7,
        bad_threshold: float = 0.4,
        output_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Export preference pairs for RLHF/DPO training.

        For each query, finds good and bad answers to create preference pairs.

        Args:
            good_threshold: Minimum score for "chosen" answer
            bad_threshold: Maximum score for "rejected" answer
            output_path: Optional file path to save JSON

        Returns:
            List of preference pairs
        """
        cursor = self.conn.cursor()

        # Find queries with both good and bad answers
        cursor.execute('''
            SELECT DISTINCT a1.query
            FROM answers a1
            INNER JOIN quality_ratings q1 ON a1.record_id = q1.record_id
            WHERE EXISTS (
                SELECT 1 FROM answers a2
                INNER JOIN quality_ratings q2 ON a2.record_id = q2.record_id
                WHERE a2.query = a1.query
                AND q2.overall_score >= ?
            )
            AND EXISTS (
                SELECT 1 FROM answers a3
                INNER JOIN quality_ratings q3 ON a3.record_id = q3.record_id
                WHERE a3.query = a1.query
                AND q3.overall_score < ?
            )
        ''', (good_threshold, bad_threshold))

        queries = [row[0] for row in cursor.fetchall()]

        pairs = []
        for query in queries:
            # Get best answer for this query
            cursor.execute('''
                SELECT a.*, q.* FROM answers a
                INNER JOIN quality_ratings q ON a.record_id = q.record_id
                WHERE a.query = ? AND q.overall_score >= ?
                ORDER BY q.overall_score DESC
                LIMIT 1
            ''', (query, good_threshold))
            good_row = cursor.fetchone()

            # Get worst answer for this query
            cursor.execute('''
                SELECT a.*, q.* FROM answers a
                INNER JOIN quality_ratings q ON a.record_id = q.record_id
                WHERE a.query = ? AND q.overall_score < ?
                ORDER BY q.overall_score ASC
                LIMIT 1
            ''', (query, bad_threshold))
            bad_row = cursor.fetchone()

            if good_row and bad_row:
                good_record = self._row_to_record(good_row)
                bad_record = self._row_to_record(bad_row)

                pairs.append({
                    'prompt': query,
                    'chosen': good_record.answer_narrative,
                    'rejected': bad_record.answer_narrative,
                    'chosen_score': good_record.quality.overall_score,
                    'rejected_score': bad_record.quality.overall_score,
                    'score_margin': good_record.quality.overall_score - bad_record.quality.overall_score
                })

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(pairs, f, indent=2)

        return pairs

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        cursor = self.conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM answers')
        total_answers = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM quality_ratings')
        rated_answers = cursor.fetchone()[0]

        cursor.execute('SELECT AVG(overall_score) FROM quality_ratings')
        avg_score = cursor.fetchone()[0] or 0

        cursor.execute('SELECT COUNT(*) FROM quality_ratings WHERE overall_score >= 0.7')
        good_answers = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM quality_ratings WHERE overall_score < 0.4')
        bad_answers = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(DISTINCT model_used) FROM answers')
        unique_models = cursor.fetchone()[0]

        return {
            'total_answers': total_answers,
            'rated_answers': rated_answers,
            'unrated_answers': total_answers - rated_answers,
            'average_quality_score': round(avg_score, 3),
            'good_answers': good_answers,
            'bad_answers': bad_answers,
            'acceptable_answers': rated_answers - good_answers - bad_answers,
            'unique_models': unique_models,
            'good_answer_percentage': round(good_answers / rated_answers * 100, 1) if rated_answers > 0 else 0
        }

    def _generate_record_id(self, query: str, answer: str) -> str:
        """Generate unique record ID from query and answer."""
        content = f"{query}:{answer}:{datetime.utcnow().isoformat()}"
        hash_value = hashlib.sha256(content.encode()).hexdigest()
        return f"ans_{hash_value[:16]}"

    def _auto_rate_answer(self, record: AnswerRecord) -> AnswerQuality:
        """
        Automatically generate quality rating using heuristics.

        In production, this could use:
        - LLM-as-judge for quality assessment
        - Automatic fact-checking against knowledge base
        - Consistency checks with previous answers
        - Citation verification
        """
        # Simple heuristic-based rating for now

        # Length-based completeness
        word_count = len(record.answer_narrative.split())
        completeness = min(word_count / 200, 1.0)  # Optimal ~200 words

        # Structure-based clarity
        has_structure = len(record.answer_structured) > 0
        clarity = 0.8 if has_structure else 0.5

        # Module diversity (more modules = more rigorous)
        module_count = len(record.modules_invoked)
        analytical_rigor = min(module_count / 3, 1.0) * 0.9

        # Default values for unknowns
        factual_accuracy = 0.75  # Assume mostly accurate
        source_reliability = 0.70  # Moderate confidence in sources
        analyst_usefulness = 0.72
        actionability = 0.65
        confidence_appropriate = 0.70

        # Check for confidence markers in answer
        if any(word in record.answer_narrative.lower() for word in ['likely', 'probably', 'possibly']):
            confidence_appropriate = 0.85  # Good uncertainty expression

        quality = AnswerQuality(
            factual_accuracy=factual_accuracy,
            analytical_rigor=analytical_rigor,
            source_reliability=source_reliability,
            completeness=completeness,
            clarity=clarity,
            analyst_usefulness=analyst_usefulness,
            actionability=actionability,
            confidence_appropriate=confidence_appropriate,
            rated_by="auto",
            rated_at=datetime.utcnow()
        )

        return quality

    def _row_to_record(self, row: Tuple) -> AnswerRecord:
        """Convert database row to AnswerRecord."""
        # Parse JSON fields
        query_intent = json.loads(row[2]) if row[2] else {}
        answer_structured = json.loads(row[5]) if row[5] else {}
        modules_invoked = json.loads(row[8]) if row[8] else []
        tags = json.loads(row[11]) if row[11] else []
        metadata = json.loads(row[12]) if row[12] else {}

        record = AnswerRecord(
            record_id=row[0],
            query=row[1],
            query_intent=query_intent,
            query_timestamp=datetime.fromisoformat(row[3]),
            answer_narrative=row[4],
            answer_structured=answer_structured,
            answer_timestamp=datetime.fromisoformat(row[6]),
            model_used=row[7] or "",
            modules_invoked=modules_invoked,
            processing_time_seconds=row[9] or 0.0,
            session_id=row[10] or "",
            analyst_id=row[11] or "",
            tags=tags,
            metadata=metadata,
            version=row[13] or 1,
            parent_record_id=row[14]
        )

        # Add quality if present (columns 16+)
        if len(row) > 16 and row[16] is not None:  # factual_accuracy exists
            quality = AnswerQuality(
                factual_accuracy=row[16],
                analytical_rigor=row[17],
                source_reliability=row[18],
                completeness=row[19],
                clarity=row[20],
                analyst_usefulness=row[21],
                actionability=row[22],
                confidence_appropriate=row[23],
                overall_score=row[24],
                strengths=json.loads(row[25]) if row[25] else [],
                weaknesses=json.loads(row[26]) if row[26] else [],
                feedback_notes=row[27] or "",
                rated_by=row[28] or "auto",
                rated_at=datetime.fromisoformat(row[29]) if row[29] else None
            )
            record.quality = quality

        return record

    def close(self):
        """Close database connection."""
        self.conn.close()
