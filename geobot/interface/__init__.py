"""
GeoBotv1 Interface Layer

Operational modes for interacting with the forecasting system:
- Interactive Analyst Mode: Natural language Q&A with LLM integration
- Watch Mode: Live monitoring with automated alerts
- Replay Mode: Historical analysis and backtesting

Integrates LLM (Mistral, GPT, Claude) for natural language understanding.
"""

from .mode_manager import ModeManager, OperationalMode
from .analyst_agent import AnalystAgent, AnalysisResult
from .watch_daemon import WatchDaemon, Alert, AlertLevel
from .replay import ReplayAnalyzer, HistoricalSnapshot

__all__ = [
    "ModeManager",
    "OperationalMode",
    "AnalystAgent",
    "AnalysisResult",
    "WatchDaemon",
    "Alert",
    "AlertLevel",
    "ReplayAnalyzer",
    "HistoricalSnapshot",
]
