"""
Forecast Logging and Drift Tracking

Logs all forecast simulations to database to track probability drift over time
with associated news context.
"""

import sqlite3
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

try:
    from ..discord_bot.forecaster import ConflictForecast
    HAS_FORECASTER = True
except ImportError:
    HAS_FORECASTER = False
    ConflictForecast = None


class ForecastLogger:
    """
    Logs forecasts to SQLite database for drift tracking.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize forecast logger.

        Parameters
        ----------
        db_path : Optional[str]
            Path to SQLite database (default: ./geobot_forecasts.db)
        """
        self.db_path = db_path or str(Path.cwd() / "geobot_forecasts.db")
        self.conn = None
        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()

        # Create forecasts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                conflict_name TEXT NOT NULL,
                escalation_probability REAL NOT NULL,
                regime_change_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                confidence REAL NOT NULL,
                timeframe TEXT NOT NULL,
                key_factors TEXT,
                news_context TEXT,
                news_article_count INTEGER DEFAULT 0
            )
        ''')

        # Create news articles table (for context)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                forecast_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                link TEXT,
                summary TEXT,
                FOREIGN KEY (forecast_id) REFERENCES forecasts(id)
            )
        ''')

        # Create drift tracking view
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS forecast_drift AS
            SELECT
                conflict_name,
                timestamp,
                escalation_probability,
                regime_change_probability,
                LAG(escalation_probability) OVER (
                    PARTITION BY conflict_name ORDER BY timestamp
                ) as prev_escalation,
                LAG(regime_change_probability) OVER (
                    PARTITION BY conflict_name ORDER BY timestamp
                ) as prev_regime_change,
                news_article_count
            FROM forecasts
            ORDER BY conflict_name, timestamp
        ''')

        self.conn.commit()

    def log_forecast(self,
                    forecast: 'ConflictForecast',
                    news_articles: Optional[List[Dict[str, Any]]] = None):
        """
        Log a forecast to the database.

        Parameters
        ----------
        forecast : ConflictForecast
            Forecast to log
        news_articles : Optional[List[Dict[str, Any]]]
            News articles that informed this forecast
        """
        if not self.conn:
            self._init_database()

        cursor = self.conn.cursor()

        # Insert forecast
        cursor.execute('''
            INSERT INTO forecasts (
                timestamp, conflict_name, escalation_probability,
                regime_change_probability, risk_level, confidence,
                timeframe, key_factors, news_context, news_article_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            forecast.conflict_name,
            forecast.escalation_probability,
            forecast.regime_change_probability,
            forecast.risk_level,
            forecast.confidence,
            forecast.timeframe,
            json.dumps(forecast.key_factors),
            json.dumps(news_articles[:10]) if news_articles else None,
            len(news_articles) if news_articles else 0
        ))

        forecast_id = cursor.lastrowid

        # Insert associated news articles
        if news_articles:
            for article in news_articles[:20]:  # Limit to 20 articles
                cursor.execute('''
                    INSERT INTO news_articles (
                        forecast_id, timestamp, title, source, link, summary
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    forecast_id,
                    datetime.now().isoformat(),
                    article.get('title', ''),
                    article.get('source', ''),
                    article.get('link', ''),
                    article.get('summary', '')
                ))

        self.conn.commit()

        return forecast_id

    def get_drift_analysis(self, conflict_name: str, days: int = 7) -> Dict[str, Any]:
        """
        Analyze forecast drift for a conflict over time.

        Parameters
        ----------
        conflict_name : str
            Conflict to analyze
        days : int
            Number of days to look back (default: 7)

        Returns
        -------
        Dict[str, Any]
            Drift analysis with statistics
        """
        if not self.conn:
            self._init_database()

        cursor = self.conn.cursor()

        # Get recent forecasts
        cursor.execute('''
            SELECT
                timestamp,
                escalation_probability,
                regime_change_probability,
                risk_level,
                news_article_count
            FROM forecasts
            WHERE conflict_name = ?
            AND datetime(timestamp) >= datetime('now', '-' || ? || ' days')
            ORDER BY timestamp DESC
        ''', (conflict_name, days))

        rows = cursor.fetchall()

        if not rows:
            return {
                'conflict': conflict_name,
                'found': False,
                'message': f'No forecasts found for {conflict_name} in last {days} days'
            }

        # Calculate drift statistics
        escalation_probs = [row[1] for row in rows]
        regime_probs = [row[2] for row in rows]

        escalation_drift = max(escalation_probs) - min(escalation_probs)
        regime_drift = max(regime_probs) - min(regime_probs)

        # Get trend direction
        if len(escalation_probs) >= 2:
            esc_trend = "increasing" if escalation_probs[0] > escalation_probs[-1] else "decreasing"
            reg_trend = "increasing" if regime_probs[0] > regime_probs[-1] else "decreasing"
        else:
            esc_trend = "stable"
            reg_trend = "stable"

        return {
            'conflict': conflict_name,
            'found': True,
            'period_days': days,
            'forecast_count': len(rows),
            'latest_escalation': escalation_probs[0],
            'latest_regime_change': regime_probs[0],
            'escalation_drift': escalation_drift,
            'regime_change_drift': regime_drift,
            'escalation_trend': esc_trend,
            'regime_change_trend': reg_trend,
            'escalation_range': (min(escalation_probs), max(escalation_probs)),
            'regime_change_range': (min(regime_probs), max(regime_probs)),
            'avg_news_articles': sum(row[4] for row in rows) / len(rows)
        }

    def get_recent_forecasts(self, conflict_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent forecasts for a conflict.

        Parameters
        ----------
        conflict_name : str
            Conflict name
        limit : int
            Number of forecasts to retrieve

        Returns
        -------
        List[Dict[str, Any]]
            Recent forecasts
        """
        if not self.conn:
            self._init_database()

        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT
                id, timestamp, escalation_probability,
                regime_change_probability, risk_level, confidence,
                key_factors, news_article_count
            FROM forecasts
            WHERE conflict_name = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (conflict_name, limit))

        rows = cursor.fetchall()

        forecasts = []
        for row in rows:
            forecasts.append({
                'id': row[0],
                'timestamp': row[1],
                'escalation_probability': row[2],
                'regime_change_probability': row[3],
                'risk_level': row[4],
                'confidence': row[5],
                'key_factors': json.loads(row[6]) if row[6] else [],
                'news_article_count': row[7]
            })

        return forecasts

    def get_all_conflicts(self) -> List[str]:
        """
        Get list of all conflicts in database.

        Returns
        -------
        List[str]
            List of conflict names
        """
        if not self.conn:
            self._init_database()

        cursor = self.conn.cursor()
        cursor.execute('SELECT DISTINCT conflict_name FROM forecasts ORDER BY conflict_name')

        return [row[0] for row in cursor.fetchall()]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall database statistics.

        Returns
        -------
        Dict[str, Any]
            Statistics
        """
        if not self.conn:
            self._init_database()

        cursor = self.conn.cursor()

        # Total forecasts
        cursor.execute('SELECT COUNT(*) FROM forecasts')
        total_forecasts = cursor.fetchone()[0]

        # Total conflicts
        cursor.execute('SELECT COUNT(DISTINCT conflict_name) FROM forecasts')
        total_conflicts = cursor.fetchone()[0]

        # Total news articles
        cursor.execute('SELECT COUNT(*) FROM news_articles')
        total_news = cursor.fetchone()[0]

        # Most forecasted conflict
        cursor.execute('''
            SELECT conflict_name, COUNT(*) as count
            FROM forecasts
            GROUP BY conflict_name
            ORDER BY count DESC
            LIMIT 1
        ''')
        top_conflict = cursor.fetchone()

        return {
            'total_forecasts': total_forecasts,
            'total_conflicts': total_conflicts,
            'total_news_articles': total_news,
            'most_forecasted_conflict': top_conflict[0] if top_conflict else None,
            'most_forecast_count': top_conflict[1] if top_conflict else 0,
            'database_path': self.db_path
        }

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# Global logger instance
_global_logger = None


def get_logger(db_path: Optional[str] = None) -> ForecastLogger:
    """
    Get global forecast logger instance.

    Parameters
    ----------
    db_path : Optional[str]
        Database path (uses default if not specified)

    Returns
    -------
    ForecastLogger
        Forecast logger instance
    """
    global _global_logger

    if _global_logger is None:
        _global_logger = ForecastLogger(db_path)

    return _global_logger
