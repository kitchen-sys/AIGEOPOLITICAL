"""
Machine Learning enhancers for GeoBotv1

Optional but powerful additions once the causal backbone is built.
Critical principle: These help discover new relationships but must not replace causality.
"""

from .risk_scoring import RiskScorer
from .feature_discovery import FeatureDiscovery
from .embedding import GeopoliticalEmbedding
# GNN imports are optional (require PyTorch)
try:
    from .graph_neural_networks import (
        CausalGNN,
        GeopoliticalNetworkGNN,
        AttentionGNN,
        MessagePassingCausalGNN,
        GNNTrainer,
        NetworkToGraph
    )
    _has_gnn = True
except ImportError:
    _has_gnn = False

__all__ = [
    "RiskScorer",
    "FeatureDiscovery",
    "GeopoliticalEmbedding",
]

if _has_gnn:
    __all__.extend([
        "CausalGNN",
        "GeopoliticalNetworkGNN",
        "AttentionGNN",
        "MessagePassingCausalGNN",
        "GNNTrainer",
        "NetworkToGraph",
    ])
