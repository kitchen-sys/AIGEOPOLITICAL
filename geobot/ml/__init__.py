"""
Machine Learning enhancers for GeoBotv1

Optional but powerful additions once the causal backbone is built.
Critical principle: These help discover new relationships but must not replace causality.
"""

from .risk_scoring import RiskScorer
from .feature_discovery import FeatureDiscovery
from .embedding import GeopoliticalEmbedding

__all__ = [
    "RiskScorer",
    "FeatureDiscovery",
    "GeopoliticalEmbedding",
]
