"""
Replay Mode for GeoBotv1 - Forensic Analysis and Backtesting

Allows answering questions like:
- "How did your forecast for X change between March and June?"
- "What would have happened if we had intervened in April?"
- "Show me the progression of risk scores over Q1 2024"
- "Why did you predict escalation when it didn't happen?"

Features:
- Reconstruct model state at any historical point
- Replay events chronologically
- Backtest forecasts against actuals
- Counterfactual "what if" scenarios on historical data
- Model performance analysis over time
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path


@dataclass
class HistoricalSnapshot:
    """Snapshot of system state at a point in time."""
    timestamp: datetime
    model_states: Dict[str, Any]  # Model parameters at this time
    risk_scores: Dict[str, float]  # Risk assessments
    forecasts: Dict[str, Any]  # Active forecasts
    events: List[str]  # Event IDs that occurred
    alerts: List[str]  # Alert IDs generated
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Results from backtesting a forecast."""
    forecast_date: datetime
    forecast_horizon: int  # days
    target_date: datetime
    predicted: Dict[str, float]
    actual: Dict[str, float]
    errors: Dict[str, float]
    metrics: Dict[str, float]  # MAE, MSE, etc.


class ReplayAnalyzer:
    """
    Replay and forensic analysis system.

    Example:
        >>> replay = ReplayAnalyzer(history_path="model_history.db")
        >>>
        >>> # Reconstruct state on a specific date
        >>> snapshot = replay.get_snapshot(datetime(2024, 3, 15))
        >>> print(f"Risk score on 3/15: {snapshot.risk_scores['iran']}")
        >>>
        >>> # Show forecast evolution
        >>> evolution = replay.track_forecast_evolution(
        ...     topic="iran_nuclear",
        ...     start_date=datetime(2024, 1, 1),
        ...     end_date=datetime(2024, 6, 30)
        ... )
        >>>
        >>> # Backtest historical forecasts
        >>> results = replay.backtest_forecasts(
        ...     start_date=datetime(2024, 1, 1),
        ...     end_date=datetime(2024, 6, 30),
        ...     horizon=30
        ... )
        >>>
        >>> # Counterfactual: What if sanctions were imposed earlier?
        >>> counterfactual = replay.run_counterfactual(
        ...     intervention_date=datetime(2024, 2, 1),
        ...     intervention={"sanctions": 1.0},
        ...     actual_date=datetime(2024, 5, 1)
        ... )
    """

    def __init__(self, history_path: str = "model_history.db"):
        """
        Initialize replay analyzer.

        Args:
            history_path: Path to historical state database
        """
        self.history_path = Path(history_path)
        self.snapshots: List[HistoricalSnapshot] = []

        # Load historical snapshots
        self._load_history()

    def get_snapshot(self, target_date: datetime) -> Optional[HistoricalSnapshot]:
        """
        Get system state at a specific date.

        Args:
            target_date: Date to retrieve

        Returns:
            HistoricalSnapshot or None if not available
        """
        # Find closest snapshot before or at target date
        valid_snapshots = [s for s in self.snapshots if s.timestamp <= target_date]

        if not valid_snapshots:
            return None

        # Return closest
        closest = max(valid_snapshots, key=lambda s: s.timestamp)
        return closest

    def get_snapshots_in_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[HistoricalSnapshot]:
        """Get all snapshots in a date range."""
        return [
            s for s in self.snapshots
            if start_date <= s.timestamp <= end_date
        ]

    def track_forecast_evolution(
        self,
        topic: str,
        start_date: datetime,
        end_date: datetime,
        granularity: str = "daily"
    ) -> List[Dict[str, Any]]:
        """
        Track how forecasts for a topic changed over time.

        Args:
            topic: Topic/entity to track (e.g., "iran_nuclear")
            start_date: Start of tracking period
            end_date: End of tracking period
            granularity: "daily", "weekly", or "monthly"

        Returns:
            List of forecast snapshots showing evolution
        """
        snapshots = self.get_snapshots_in_range(start_date, end_date)

        evolution = []
        for snapshot in snapshots:
            if topic in snapshot.risk_scores:
                evolution.append({
                    'date': snapshot.timestamp,
                    'risk_score': snapshot.risk_scores[topic],
                    'forecasts': snapshot.forecasts.get(topic, {}),
                    'events_count': len(snapshot.events)
                })

        return evolution

    def backtest_forecasts(
        self,
        start_date: datetime,
        end_date: datetime,
        horizon: int = 30,  # days
        metrics: Optional[List[str]] = None
    ) -> List[BacktestResult]:
        """
        Backtest historical forecasts against actual outcomes.

        Args:
            start_date: Start of backtest period
            end_date: End of backtest period
            horizon: Forecast horizon in days
            metrics: Metrics to compute ('mae', 'mse', 'accuracy')

        Returns:
            List of backtest results
        """
        if metrics is None:
            metrics = ['mae', 'mse']

        results = []
        snapshots = self.get_snapshots_in_range(start_date, end_date)

        for snapshot in snapshots:
            target_date = snapshot.timestamp + timedelta(days=horizon)

            # Get actual outcome at target date
            actual_snapshot = self.get_snapshot(target_date)
            if not actual_snapshot:
                continue

            # Compare forecast vs actual
            predicted = snapshot.forecasts
            actual = actual_snapshot.risk_scores

            # Compute errors
            errors = {}
            for key in set(predicted.keys()) & set(actual.keys()):
                errors[key] = abs(predicted[key] - actual[key])

            # Compute metrics
            computed_metrics = {}
            if 'mae' in metrics and errors:
                computed_metrics['mae'] = sum(errors.values()) / len(errors)
            if 'mse' in metrics and errors:
                computed_metrics['mse'] = sum(e**2 for e in errors.values()) / len(errors)

            results.append(BacktestResult(
                forecast_date=snapshot.timestamp,
                forecast_horizon=horizon,
                target_date=target_date,
                predicted=predicted,
                actual=actual,
                errors=errors,
                metrics=computed_metrics
            ))

        return results

    def run_counterfactual(
        self,
        intervention_date: datetime,
        intervention: Dict[str, float],
        actual_date: datetime
    ) -> Dict[str, Any]:
        """
        Run counterfactual "what if" analysis on historical data.

        Args:
            intervention_date: When hypothetical intervention occurs
            intervention: Intervention to simulate
            actual_date: Date to compare against

        Returns:
            Comparison of actual vs counterfactual outcomes
        """
        # Get pre-intervention state
        pre_intervention = self.get_snapshot(intervention_date)
        if not pre_intervention:
            return {"error": "No snapshot available for intervention date"}

        # Get actual outcome
        actual_outcome = self.get_snapshot(actual_date)
        if not actual_outcome:
            return {"error": "No snapshot available for actual date"}

        # Simulate counterfactual (simplified - in production, rerun models)
        counterfactual_risk = {}
        for key, actual_risk in actual_outcome.risk_scores.items():
            # Estimate intervention effect (placeholder)
            effect = intervention.get('effect_multiplier', 0.8)
            counterfactual_risk[key] = actual_risk * effect

        return {
            'intervention_date': intervention_date.isoformat(),
            'actual_date': actual_date.isoformat(),
            'intervention': intervention,
            'actual_outcomes': actual_outcome.risk_scores,
            'counterfactual_outcomes': counterfactual_risk,
            'differences': {
                k: actual_outcome.risk_scores[k] - counterfactual_risk[k]
                for k in counterfactual_risk
            }
        }

    def analyze_forecast_accuracy(
        self,
        start_date: datetime,
        end_date: datetime,
        horizons: List[int] = [7, 30, 90]
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of forecast accuracy over time.

        Args:
            start_date: Analysis start
            end_date: Analysis end
            horizons: Forecast horizons to analyze

        Returns:
            Accuracy analysis by horizon
        """
        analysis = {}

        for horizon in horizons:
            backtest_results = self.backtest_forecasts(
                start_date, end_date, horizon=horizon
            )

            if backtest_results:
                mae_values = [r.metrics.get('mae', float('nan')) for r in backtest_results]
                mae_values = [v for v in mae_values if not np.isnan(v)]

                analysis[f"{horizon}d"] = {
                    'horizon_days': horizon,
                    'n_forecasts': len(backtest_results),
                    'mean_mae': np.mean(mae_values) if mae_values else None,
                    'median_mae': np.median(mae_values) if mae_values else None,
                    'forecast_skill': self._compute_forecast_skill(backtest_results)
                }

        return analysis

    def explain_forecast_change(
        self,
        topic: str,
        date1: datetime,
        date2: datetime
    ) -> Dict[str, Any]:
        """
        Explain why forecast changed between two dates.

        Args:
            topic: Topic to analyze
            date1: Earlier date
            date2: Later date

        Returns:
            Explanation of changes
        """
        snapshot1 = self.get_snapshot(date1)
        snapshot2 = self.get_snapshot(date2)

        if not snapshot1 or not snapshot2:
            return {"error": "Missing snapshots"}

        # Compare risk scores
        risk1 = snapshot1.risk_scores.get(topic, 0)
        risk2 = snapshot2.risk_scores.get(topic, 0)
        risk_change = risk2 - risk1

        # Events between dates
        events_between = []
        for s in self.get_snapshots_in_range(date1, date2):
            events_between.extend(s.events)

        # Model state changes
        model_changes = {}
        for model_name in set(snapshot1.model_states.keys()) | set(snapshot2.model_states.keys()):
            state1 = snapshot1.model_states.get(model_name, {})
            state2 = snapshot2.model_states.get(model_name, {})
            if state1 != state2:
                model_changes[model_name] = {
                    'before': state1,
                    'after': state2
                }

        return {
            'topic': topic,
            'date1': date1.isoformat(),
            'date2': date2.isoformat(),
            'risk_change': risk_change,
            'risk_change_pct': (risk_change / risk1 * 100) if risk1 != 0 else None,
            'events_between': events_between,
            'n_events': len(events_between),
            'model_changes': model_changes,
            'explanation': self._generate_change_explanation(risk_change, events_between, model_changes)
        }

    def generate_performance_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> str:
        """
        Generate comprehensive performance report.

        Args:
            start_date: Report start
            end_date: Report end

        Returns:
            Formatted report string
        """
        # Accuracy analysis
        accuracy = self.analyze_forecast_accuracy(start_date, end_date)

        # Backtest results
        backtest = self.backtest_forecasts(start_date, end_date, horizon=30)

        report_lines = []
        report_lines.append("="*70)
        report_lines.append("GeoBotv1 - Historical Performance Report")
        report_lines.append("="*70)
        report_lines.append(f"\nPeriod: {start_date.date()} to {end_date.date()}")
        report_lines.append(f"Duration: {(end_date - start_date).days} days\n")

        report_lines.append("📊 Forecast Accuracy by Horizon:")
        report_lines.append("-"*70)
        for horizon_name, metrics in accuracy.items():
            report_lines.append(f"\n{horizon_name} Forecasts:")
            report_lines.append(f"  • Number of forecasts: {metrics['n_forecasts']}")
            if metrics['mean_mae'] is not None:
                report_lines.append(f"  • Mean Absolute Error: {metrics['mean_mae']:.3f}")
                report_lines.append(f"  • Median Absolute Error: {metrics['median_mae']:.3f}")
            if metrics['forecast_skill'] is not None:
                report_lines.append(f"  • Forecast Skill: {metrics['forecast_skill']:.2%}")

        report_lines.append("\n📈 Backtest Summary (30-day forecasts):")
        report_lines.append("-"*70)
        if backtest:
            report_lines.append(f"Total forecasts tested: {len(backtest)}")
            all_errors = [e for r in backtest for e in r.errors.values()]
            if all_errors:
                report_lines.append(f"Overall MAE: {np.mean(all_errors):.3f}")
                report_lines.append(f"Best forecast MAE: {min(all_errors):.3f}")
                report_lines.append(f"Worst forecast MAE: {max(all_errors):.3f}")

        report_lines.append("\n" + "="*70)

        return '\n'.join(report_lines)

    def _load_history(self) -> None:
        """Load historical snapshots from storage."""
        if not self.history_path.exists():
            # Create dummy history for demonstration
            self._create_dummy_history()
            return

        # In production, load from database
        # For now, use dummy data
        self._create_dummy_history()

    def _create_dummy_history(self) -> None:
        """Create dummy historical data for demonstration."""
        import random

        start_date = datetime(2024, 1, 1)
        for day in range(180):  # 6 months of daily snapshots
            date = start_date + timedelta(days=day)

            snapshot = HistoricalSnapshot(
                timestamp=date,
                model_states={
                    'var': {'coefficients': [[0.5, 0.2], [0.1, 0.6]]},
                    'hawkes': {'branching_ratio': random.uniform(0.4, 0.8)}
                },
                risk_scores={
                    'iran': random.uniform(0.4, 0.8),
                    'russia': random.uniform(0.3, 0.7),
                    'china': random.uniform(0.2, 0.6)
                },
                forecasts={
                    'iran': random.uniform(0.5, 0.9),
                    'russia': random.uniform(0.4, 0.8),
                    'china': random.uniform(0.3, 0.7)
                },
                events=[f"event_{day}_{i}" for i in range(random.randint(0, 5))],
                alerts=[]
            )

            self.snapshots.append(snapshot)

    def _compute_forecast_skill(self, results: List[BacktestResult]) -> Optional[float]:
        """
        Compute forecast skill score.

        Skill > 0: Better than persistence forecast
        Skill = 0: Same as persistence
        Skill < 0: Worse than persistence
        """
        if not results:
            return None

        # Simplified skill score
        total_error = sum(sum(r.errors.values()) for r in results if r.errors)
        n_forecasts = sum(len(r.errors) for r in results)

        if n_forecasts == 0:
            return None

        mean_error = total_error / n_forecasts

        # Compare to naive persistence forecast (assume 0.5 as baseline error)
        baseline_error = 0.5
        skill = 1 - (mean_error / baseline_error)

        return skill

    def _generate_change_explanation(
        self,
        risk_change: float,
        events: List[str],
        model_changes: Dict
    ) -> str:
        """Generate natural language explanation of forecast change."""
        direction = "increased" if risk_change > 0 else "decreased"
        magnitude = abs(risk_change)

        parts = []
        parts.append(f"Risk {direction} by {magnitude:.2f}")

        if events:
            parts.append(f"driven by {len(events)} new events in the period")

        if model_changes:
            parts.append(f"and {len(model_changes)} model parameter updates")

        return ". ".join(parts) + "."


# Import numpy for statistics
import numpy as np
