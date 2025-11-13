"""
Time-series modeling and state-space models for GeoBotv1
"""

from .kalman_filter import KalmanFilter, ExtendedKalmanFilter
from .hmm import HiddenMarkovModel
from .regime_switching import RegimeSwitchingModel
from .var_models import (
    VARModel,
    SVARModel,
    DynamicFactorModel,
    GrangerCausality,
    VARResults,
    IRFResult
)
from .point_processes import (
    UnivariateHawkesProcess,
    MultivariateHawkesProcess,
    ConflictContagionModel,
    HawkesParameters,
    HawkesFitResult,
    estimate_branching_ratio,
    detect_explosive_regime
)

__all__ = [
    "KalmanFilter",
    "ExtendedKalmanFilter",
    "HiddenMarkovModel",
    "RegimeSwitchingModel",
    "VARModel",
    "SVARModel",
    "DynamicFactorModel",
    "GrangerCausality",
    "VARResults",
    "IRFResult",
    "UnivariateHawkesProcess",
    "MultivariateHawkesProcess",
    "ConflictContagionModel",
    "HawkesParameters",
    "HawkesFitResult",
    "estimate_branching_ratio",
    "detect_explosive_regime",
]
