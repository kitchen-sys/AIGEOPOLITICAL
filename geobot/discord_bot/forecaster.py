"""
Conflict Forecasting for Discord Bot

Provides real-time escalation and regime change probability forecasts
using Bayesian inference and GeoBot 2.0 analytical framework.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    from ..bayes.forecasting import BayesianForecaster, PriorType, GeopoliticalPrior
    HAS_BAYES = True
except ImportError:
    HAS_BAYES = False

try:
    from ..analysis.engine import AnalyticalEngine
    HAS_ANALYSIS = True
except ImportError:
    HAS_ANALYSIS = False


@dataclass
class ConflictForecast:
    """
    Forecast for a specific conflict scenario.

    Attributes
    ----------
    conflict_name : str
        Name of the conflict/region
    escalation_probability : float
        Probability of escalation (0-1)
    regime_change_probability : float
        Probability of regime change (0-1)
    risk_level : str
        Overall risk assessment (critical/high/medium/low)
    key_factors : List[str]
        Key factors influencing the forecast
    confidence : float
        Forecast confidence level (0-1)
    timeframe : str
        Forecast timeframe (e.g., "3 months", "6 months")
    """
    conflict_name: str
    escalation_probability: float
    regime_change_probability: float
    risk_level: str
    key_factors: List[str]
    confidence: float
    timeframe: str


class ConflictForecaster:
    """
    Forecasts conflict escalation and regime change probabilities.

    Uses Bayesian priors, keyword analysis, and GeoBot 2.0 analytics.
    """

    # Predefined conflict scenarios with baseline priors
    CONFLICT_PRIORS = {
        'taiwan': {
            'escalation': {'alpha': 3.0, 'beta': 17.0},  # ~15% base rate
            'regime_change': {'alpha': 1.5, 'beta': 18.5},  # ~7.5% base rate
            'timeframe': '12 months'
        },
        'ukraine': {
            'escalation': {'alpha': 8.0, 'beta': 12.0},  # ~40% base rate
            'regime_change': {'alpha': 4.0, 'beta': 16.0},  # ~20% base rate
            'timeframe': '6 months'
        },
        'iran': {
            'escalation': {'alpha': 4.0, 'beta': 16.0},  # ~20% base rate
            'regime_change': {'alpha': 3.0, 'beta': 17.0},  # ~15% base rate
            'timeframe': '12 months'
        },
        'north_korea': {
            'escalation': {'alpha': 5.0, 'beta': 15.0},  # ~25% base rate
            'regime_change': {'alpha': 2.0, 'beta': 18.0},  # ~10% base rate
            'timeframe': '12 months'
        },
        'israel_palestine': {
            'escalation': {'alpha': 7.0, 'beta': 13.0},  # ~35% base rate
            'regime_change': {'alpha': 2.5, 'beta': 17.5},  # ~12.5% base rate
            'timeframe': '6 months'
        },
        'syria': {
            'escalation': {'alpha': 6.0, 'beta': 14.0},  # ~30% base rate
            'regime_change': {'alpha': 5.0, 'beta': 15.0},  # ~25% base rate
            'timeframe': '12 months'
        },
        'kashmir': {
            'escalation': {'alpha': 4.5, 'beta': 15.5},  # ~22.5% base rate
            'regime_change': {'alpha': 1.0, 'beta': 19.0},  # ~5% base rate
            'timeframe': '12 months'
        },
        'venezuela': {
            'escalation': {'alpha': 3.5, 'beta': 16.5},  # ~17.5% base rate
            'regime_change': {'alpha': 6.0, 'beta': 14.0},  # ~30% base rate (higher due to internal instability)
            'timeframe': '12 months'
        },
        'usa_venezuela': {
            'escalation': {'alpha': 2.5, 'beta': 17.5},  # ~12.5% base rate (US intervention)
            'regime_change': {'alpha': 6.5, 'beta': 13.5},  # ~32.5% base rate (regime change focus)
            'timeframe': '12 months'
        }
    }

    # Escalation keywords and their weights
    ESCALATION_KEYWORDS = {
        'critical': ['war', 'invasion', 'attack', 'strike', 'nuclear', 'missile launch'],
        'high': ['military buildup', 'mobilization', 'threat', 'warning', 'sanctions', 'blockade'],
        'medium': ['tension', 'drills', 'exercises', 'patrol', 'surveillance'],
        'low': ['talks', 'dialogue', 'negotiations', 'ceasefire', 'agreement']
    }

    # Regime change keywords
    REGIME_CHANGE_KEYWORDS = {
        'high': ['coup', 'revolution', 'uprising', 'overthrow', 'collapse', 'rebellion'],
        'medium': ['protests', 'demonstrations', 'unrest', 'opposition', 'dissent'],
        'low': ['reform', 'transition', 'election', 'stability']
    }

    def __init__(self):
        """Initialize forecaster."""
        self.forecaster = BayesianForecaster() if HAS_BAYES else None
        self.engine = AnalyticalEngine() if HAS_ANALYSIS else None

    def _normalize_conflict_name(self, conflict: str) -> str:
        """Normalize conflict name to match predefined scenarios."""
        conflict_lower = conflict.lower().strip()

        # Map common variations to canonical names
        mappings = {
            'taiwan strait': 'taiwan',
            'china taiwan': 'taiwan',
            'russia ukraine': 'ukraine',
            'ukraine war': 'ukraine',
            'north korea': 'north_korea',
            'dprk': 'north_korea',
            'israel palestine': 'israel_palestine',
            'gaza': 'israel_palestine',
            'west bank': 'israel_palestine',
            'kashmir': 'kashmir',
            'india pakistan': 'kashmir',
            'usa venezuela': 'usa_venezuela',
            'us venezuela': 'usa_venezuela',
            'america venezuela': 'usa_venezuela',
            'venezuela crisis': 'venezuela',
            'maduro': 'venezuela'
        }

        for variant, canonical in mappings.items():
            if variant in conflict_lower:
                return canonical

        # Check if it matches any predefined conflict
        for known_conflict in self.CONFLICT_PRIORS.keys():
            if known_conflict.replace('_', ' ') in conflict_lower:
                return known_conflict

        # Return as-is if no match
        return conflict_lower.replace(' ', '_')

    def _assess_keywords(self, text: str, keywords_dict: Dict[str, List[str]]) -> Tuple[str, float]:
        """
        Assess text for keyword matches.

        Returns
        -------
        Tuple[str, float]
            (risk_level, adjustment_factor)
        """
        text_lower = text.lower()
        scores = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}

        for level, keywords in keywords_dict.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[level] += 1

        # Determine dominant level
        max_score = max(scores.values())
        if max_score == 0:
            return 'medium', 1.0

        for level in ['critical', 'high', 'medium', 'low']:
            if scores[level] == max_score:
                # Adjustment factor based on level
                adjustments = {
                    'critical': 1.5,
                    'high': 1.25,
                    'medium': 1.0,
                    'low': 0.75
                }
                return level, adjustments[level]

        return 'medium', 1.0

    def forecast_conflict(self,
                         conflict: str,
                         context_text: Optional[str] = None,
                         recent_articles: Optional[List[str]] = None) -> ConflictForecast:
        """
        Generate conflict forecast with escalation and regime change probabilities.

        Parameters
        ----------
        conflict : str
            Conflict name or region
        context_text : Optional[str]
            Additional context for analysis
        recent_articles : Optional[List[str]]
            Recent news article titles/summaries

        Returns
        -------
        ConflictForecast
            Forecast with probabilities and analysis
        """
        # Normalize conflict name
        normalized_conflict = self._normalize_conflict_name(conflict)

        # Get prior or use default
        if normalized_conflict in self.CONFLICT_PRIORS:
            prior_data = self.CONFLICT_PRIORS[normalized_conflict]
            timeframe = prior_data['timeframe']
        else:
            # Default priors for unknown conflicts
            prior_data = {
                'escalation': {'alpha': 3.0, 'beta': 17.0},
                'regime_change': {'alpha': 2.0, 'beta': 18.0},
                'timeframe': '12 months'
            }
            timeframe = '12 months'

        # Build analysis text
        analysis_text = f"{conflict} "
        if context_text:
            analysis_text += context_text + " "
        if recent_articles:
            analysis_text += " ".join(recent_articles)

        # Assess keywords for escalation
        escalation_level, escalation_adj = self._assess_keywords(
            analysis_text, self.ESCALATION_KEYWORDS
        )

        # Assess keywords for regime change
        regime_level, regime_adj = self._assess_keywords(
            analysis_text, self.REGIME_CHANGE_KEYWORDS
        )

        # Calculate adjusted probabilities
        if HAS_NUMPY and HAS_BAYES:
            # Use Bayesian approach
            escalation_prior = GeopoliticalPrior(
                parameter_name=f"{normalized_conflict}_escalation",
                prior_type=PriorType.BETA,
                parameters=prior_data['escalation'],
                rationale="Historical base rate"
            )

            regime_prior = GeopoliticalPrior(
                parameter_name=f"{normalized_conflict}_regime_change",
                prior_type=PriorType.BETA,
                parameters=prior_data['regime_change'],
                rationale="Historical base rate"
            )

            # Sample from posteriors (simplified - just adjust the mean)
            escalation_samples = self.forecaster.sample_prior(escalation_prior, n_samples=10000)
            regime_samples = self.forecaster.sample_prior(regime_prior, n_samples=10000)

            escalation_prob = float(np.mean(escalation_samples) * escalation_adj)
            regime_prob = float(np.mean(regime_samples) * regime_adj)

            confidence = 0.7  # Base confidence

        else:
            # Fallback: simple calculation from beta parameters
            esc_alpha, esc_beta = prior_data['escalation']['alpha'], prior_data['escalation']['beta']
            reg_alpha, reg_beta = prior_data['regime_change']['alpha'], prior_data['regime_change']['beta']

            escalation_prob = (esc_alpha / (esc_alpha + esc_beta)) * escalation_adj
            regime_prob = (reg_alpha / (reg_alpha + reg_beta)) * regime_adj

            confidence = 0.6

        # Clip probabilities to [0, 1]
        escalation_prob = max(0.0, min(1.0, escalation_prob))
        regime_prob = max(0.0, min(1.0, regime_prob))

        # Determine overall risk level
        if escalation_prob > 0.6 or escalation_level == 'critical':
            risk_level = 'critical'
        elif escalation_prob > 0.4 or escalation_level == 'high':
            risk_level = 'high'
        elif escalation_prob > 0.2:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        # Extract key factors
        key_factors = []
        if escalation_level in ['critical', 'high']:
            key_factors.append(f"Escalation indicators: {escalation_level}")
        if regime_level in ['high', 'medium']:
            key_factors.append(f"Regime stability concerns: {regime_level}")
        if context_text:
            key_factors.append("Recent developments analyzed")

        if not key_factors:
            key_factors = ["Baseline assessment", "Limited recent intelligence"]

        return ConflictForecast(
            conflict_name=conflict,
            escalation_probability=escalation_prob,
            regime_change_probability=regime_prob,
            risk_level=risk_level,
            key_factors=key_factors,
            confidence=confidence,
            timeframe=timeframe
        )

    def compare_nations(self,
                       nation1: str,
                       nation2: str,
                       context: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare two nations in conflict context.

        Parameters
        ----------
        nation1 : str
            First nation
        nation2 : str
            Second nation
        context : Optional[str]
            Conflict context

        Returns
        -------
        Dict[str, Any]
            Comparison analysis
        """
        # Build comparison using GeoBot 2.0 if available
        if self.engine:
            analysis_context = {
                'nation1': nation1,
                'nation2': nation2,
                'comparison_type': 'conflict',
                'context': context or 'general comparison'
            }

            query = f"Compare {nation1} and {nation2} in conflict context"
            analysis = self.engine.analyze(query, analysis_context)

            return {
                'nation1': nation1,
                'nation2': nation2,
                'analysis': analysis,
                'method': 'GeoBot 2.0 analytical framework'
            }
        else:
            return {
                'nation1': nation1,
                'nation2': nation2,
                'analysis': f"Comparison requested between {nation1} and {nation2}. Enable GeoBot 2.0 for detailed analysis.",
                'method': 'basic'
            }


def quick_conflict_scan(conflict: str) -> ConflictForecast:
    """
    Quick conflict scan for Discord bot.

    Parameters
    ----------
    conflict : str
        Conflict to scan

    Returns
    -------
    ConflictForecast
        Forecast with probabilities
    """
    forecaster = ConflictForecaster()
    return forecaster.forecast_conflict(conflict)
