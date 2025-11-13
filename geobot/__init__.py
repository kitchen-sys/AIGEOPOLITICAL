"""
GeoBotv1: Geopolitical Forecasting Framework

A comprehensive framework for geopolitical risk analysis, conflict prediction,
and intervention simulation using advanced mathematical and statistical methods.
"""

__version__ = "0.1.0"
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
]
