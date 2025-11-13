"""
Mode Manager for GeoBotv1 Operational Modes

Manages transitions between different operational modes:
- Interactive: Chat-based analysis with human analyst
- Watch: Autonomous monitoring with alerts
- Replay: Historical analysis and backtesting
"""

from enum import Enum
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path


class OperationalMode(Enum):
    """Operational modes for GeoBotv1."""
    INTERACTIVE = "interactive"  # Chat & Ask mode
    WATCH = "watch"              # Live monitoring mode
    REPLAY = "replay"            # Forensic/backtesting mode
    IDLE = "idle"                # Standby


@dataclass
class ModeConfig:
    """Configuration for an operational mode."""
    mode: OperationalMode
    settings: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    session_id: Optional[str] = None


@dataclass
class ModeTransition:
    """Record of mode transition."""
    from_mode: OperationalMode
    to_mode: OperationalMode
    timestamp: datetime
    reason: str
    triggered_by: str  # "user" or "system"


class ModeManager:
    """
    Manages operational modes for GeoBotv1.

    Handles:
    - Mode transitions (interactive ↔ watch ↔ replay)
    - Mode-specific configuration
    - State persistence across mode changes
    - Transition logging

    Example:
        >>> manager = ModeManager()
        >>>
        >>> # Start interactive mode
        >>> manager.set_mode(OperationalMode.INTERACTIVE, settings={
        ...     "llm_model": "mistral",
        ...     "verbosity": "detailed"
        ... })
        >>>
        >>> # Switch to watch mode
        >>> manager.set_mode(OperationalMode.WATCH, settings={
        ...     "check_interval": 300,  # 5 minutes
        ...     "alert_threshold": "medium"
        ... })
        >>>
        >>> # Enter replay mode for specific time range
        >>> manager.set_mode(OperationalMode.REPLAY, settings={
        ...     "start_date": "2024-01-01",
        ...     "end_date": "2024-06-30",
        ...     "focus": "iran_nuclear"
        ... })
    """

    def __init__(self, state_path: str = ".geobot_state.json"):
        """
        Initialize mode manager.

        Args:
            state_path: Path to persist state
        """
        self.state_path = Path(state_path)
        self.current_mode = OperationalMode.IDLE
        self.current_config: Optional[ModeConfig] = None
        self.transition_history: list[ModeTransition] = []
        self.mode_handlers: Dict[OperationalMode, Callable] = {}

        # Load previous state if exists
        self._load_state()

    def set_mode(
        self,
        mode: OperationalMode,
        settings: Optional[Dict[str, Any]] = None,
        reason: str = "manual",
        triggered_by: str = "user"
    ) -> None:
        """
        Transition to a new operational mode.

        Args:
            mode: Target mode
            settings: Mode-specific settings
            reason: Reason for transition
            triggered_by: Who/what triggered transition
        """
        # Record transition
        transition = ModeTransition(
            from_mode=self.current_mode,
            to_mode=mode,
            timestamp=datetime.utcnow(),
            reason=reason,
            triggered_by=triggered_by
        )
        self.transition_history.append(transition)

        # Update current mode
        previous_mode = self.current_mode
        self.current_mode = mode

        # Create new config
        self.current_config = ModeConfig(
            mode=mode,
            settings=settings or {},
            start_time=datetime.utcnow(),
            session_id=self._generate_session_id()
        )

        # Call mode-specific handler if registered
        if mode in self.mode_handlers:
            self.mode_handlers[mode](self.current_config)

        # Persist state
        self._save_state()

        print(f"🔄 Mode transition: {previous_mode.value} → {mode.value}")
        if reason:
            print(f"   Reason: {reason}")

    def register_mode_handler(
        self,
        mode: OperationalMode,
        handler: Callable[[ModeConfig], None]
    ) -> None:
        """
        Register a handler function to be called when entering a mode.

        Args:
            mode: Operational mode
            handler: Function to call (receives ModeConfig)
        """
        self.mode_handlers[mode] = handler

    def get_current_mode(self) -> OperationalMode:
        """Get current operational mode."""
        return self.current_mode

    def get_current_config(self) -> Optional[ModeConfig]:
        """Get current mode configuration."""
        return self.current_config

    def get_transition_history(self) -> list[ModeTransition]:
        """Get history of mode transitions."""
        return self.transition_history

    def get_mode_description(self, mode: OperationalMode) -> str:
        """Get human-readable description of a mode."""
        descriptions = {
            OperationalMode.INTERACTIVE: (
                "Interactive Analyst Mode\n"
                "• Natural language Q&A with LLM integration\n"
                "• On-demand analysis and forecasts\n"
                "• Human-in-the-loop decision support\n"
                "• Structured output: narrative + analysis blocks"
            ),
            OperationalMode.WATCH: (
                "Watch Mode (Live Monitoring)\n"
                "• Autonomous monitoring of intelligence feeds\n"
                "• Automated model updates\n"
                "• Real-time alert generation\n"
                "• Background operation with chat override"
            ),
            OperationalMode.REPLAY: (
                "Replay Mode (Forensic Analysis)\n"
                "• Historical backtesting and analysis\n"
                "• 'How did we get here?' investigations\n"
                "• Model state reconstruction\n"
                "• Counterfactual what-if scenarios"
            ),
            OperationalMode.IDLE: (
                "Idle Mode\n"
                "• System standby\n"
                "• No active monitoring or processing"
            )
        }
        return descriptions.get(mode, "Unknown mode")

    def get_mode_settings_schema(self, mode: OperationalMode) -> Dict[str, Any]:
        """Get expected settings schema for a mode."""
        schemas = {
            OperationalMode.INTERACTIVE: {
                "llm_model": {
                    "type": "string",
                    "options": ["mistral", "gpt-4", "claude-3", "local"],
                    "default": "mistral",
                    "description": "LLM model for natural language understanding"
                },
                "verbosity": {
                    "type": "string",
                    "options": ["concise", "standard", "detailed"],
                    "default": "standard",
                    "description": "Level of detail in responses"
                },
                "output_format": {
                    "type": "string",
                    "options": ["narrative", "structured", "both"],
                    "default": "both",
                    "description": "Response format"
                },
                "show_uncertainty": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include uncertainty quantification"
                }
            },
            OperationalMode.WATCH: {
                "check_interval": {
                    "type": "integer",
                    "default": 300,
                    "description": "Seconds between checks"
                },
                "alert_threshold": {
                    "type": "string",
                    "options": ["low", "medium", "high"],
                    "default": "medium",
                    "description": "Minimum alert level to notify"
                },
                "monitored_topics": {
                    "type": "list",
                    "default": [],
                    "description": "Topics to monitor"
                },
                "auto_update_models": {
                    "type": "boolean",
                    "default": True,
                    "description": "Automatically update models with new data"
                },
                "notification_channels": {
                    "type": "list",
                    "options": ["console", "email", "webhook"],
                    "default": ["console"],
                    "description": "How to send alerts"
                }
            },
            OperationalMode.REPLAY: {
                "start_date": {
                    "type": "date",
                    "required": True,
                    "description": "Start date for replay"
                },
                "end_date": {
                    "type": "date",
                    "required": True,
                    "description": "End date for replay"
                },
                "focus": {
                    "type": "string",
                    "default": None,
                    "description": "Specific topic/event to focus on"
                },
                "granularity": {
                    "type": "string",
                    "options": ["daily", "weekly", "event-driven"],
                    "default": "daily",
                    "description": "Replay granularity"
                },
                "include_counterfactuals": {
                    "type": "boolean",
                    "default": False,
                    "description": "Generate what-if scenarios"
                }
            }
        }
        return schemas.get(mode, {})

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        import uuid
        return f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

    def _save_state(self) -> None:
        """Persist current state to disk."""
        state = {
            "current_mode": self.current_mode.value,
            "current_config": {
                "mode": self.current_config.mode.value if self.current_config else None,
                "settings": self.current_config.settings if self.current_config else {},
                "start_time": self.current_config.start_time.isoformat() if self.current_config and self.current_config.start_time else None,
                "session_id": self.current_config.session_id if self.current_config else None
            } if self.current_config else None,
            "transition_history": [
                {
                    "from_mode": t.from_mode.value,
                    "to_mode": t.to_mode.value,
                    "timestamp": t.timestamp.isoformat(),
                    "reason": t.reason,
                    "triggered_by": t.triggered_by
                }
                for t in self.transition_history[-50:]  # Keep last 50
            ]
        }

        with open(self.state_path, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load state from disk."""
        if not self.state_path.exists():
            return

        try:
            with open(self.state_path, 'r') as f:
                state = json.load(f)

            # Restore current mode
            self.current_mode = OperationalMode(state.get("current_mode", "idle"))

            # Restore config
            config_data = state.get("current_config")
            if config_data:
                self.current_config = ModeConfig(
                    mode=OperationalMode(config_data["mode"]),
                    settings=config_data.get("settings", {}),
                    start_time=datetime.fromisoformat(config_data["start_time"]) if config_data.get("start_time") else None,
                    session_id=config_data.get("session_id")
                )

            # Restore transition history
            self.transition_history = [
                ModeTransition(
                    from_mode=OperationalMode(t["from_mode"]),
                    to_mode=OperationalMode(t["to_mode"]),
                    timestamp=datetime.fromisoformat(t["timestamp"]),
                    reason=t["reason"],
                    triggered_by=t["triggered_by"]
                )
                for t in state.get("transition_history", [])
            ]

        except Exception as e:
            print(f"Warning: Could not load previous state: {e}")

    def print_status(self) -> None:
        """Print current mode status."""
        print("\n" + "="*70)
        print("GeoBotv1 - Operational Status")
        print("="*70)
        print(f"\n🎯 Current Mode: {self.current_mode.value.upper()}")

        if self.current_config:
            print(f"\n📋 Configuration:")
            for key, value in self.current_config.settings.items():
                print(f"   • {key}: {value}")

            if self.current_config.start_time:
                uptime = datetime.utcnow() - self.current_config.start_time
                print(f"\n⏱️  Uptime: {uptime}")

            if self.current_config.session_id:
                print(f"🔑 Session ID: {self.current_config.session_id}")

        print(f"\n📊 Mode Transitions: {len(self.transition_history)}")
        if self.transition_history:
            print("\nRecent transitions:")
            for t in self.transition_history[-5:]:
                print(f"   {t.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: "
                      f"{t.from_mode.value} → {t.to_mode.value} ({t.reason})")

        print("\n" + "="*70)
