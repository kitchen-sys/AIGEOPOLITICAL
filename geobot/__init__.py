"""
GeoBotv1: Geopolitical Forecasting Framework

A comprehensive framework for geopolitical risk analysis, conflict prediction,
and intervention simulation using advanced mathematical and statistical methods.

Version 2.0 includes the GeoBot analytical framework for clinical systems analysis
with geopolitical nuance.
"""

__version__ = "2.0.0"
__author__ = "GeoBotv1 Team"

# Core modules
from . import core
from . import models
from . import inference
from . import simulation
from . import timeseries
from . import ml
from . import data_ingestion
from . import utils
from . import config
from . import analysis

# New modules in v2.0
from . import bayes
from . import causal

__all__ = [
    "core",
    "models",
    "inference",
    "simulation",
    "timeseries",
    "ml",
    "data_ingestion",
    "utils",
    "config",
    "analysis",
    "bayes",
    "causal",
]
