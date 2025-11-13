"""
Core mathematical frameworks for GeoBotv1
"""

from .optimal_transport import WassersteinDistance, ScenarioComparator
from .scenario import Scenario, ScenarioDistribution
from .advanced_optimal_transport import (
    GradientBasedOT,
    KantorovichDuality,
    EntropicOT,
    UnbalancedOT,
    GromovWassersteinDistance
)

__all__ = [
    "WassersteinDistance",
    "ScenarioComparator",
    "Scenario",
    "ScenarioDistribution",
    "GradientBasedOT",
    "KantorovichDuality",
    "EntropicOT",
    "UnbalancedOT",
    "GromovWassersteinDistance",
]
