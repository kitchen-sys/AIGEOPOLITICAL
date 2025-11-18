"""
Discord Bot for Real-Time Geopolitical Intelligence

Provides Discord integration for GeoBot 2.0 with:
- Auto-posting ticker updates every 5 minutes
- /compare command for nation comparison
- /scan command for conflict escalation analysis
- /ask command for geopolitical Q&A
"""

from . import bot
from . import forecaster

__all__ = ['bot', 'forecaster']
