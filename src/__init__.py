"""
Robust Demographic Circuits - Core Modules

This package contains the core functionality for demographic circuit
extraction and intervention in language models.
"""

__version__ = "0.1.0"

from .probing_classifier import AttentionHeadProber
from .intervention_engine import CircuitInterventionEngine, InterventionConfig
from .mlp_intervention_engine import MLPInterventionEngine, MLPInterventionConfig

__all__ = [
    'AttentionHeadProber',
    'CircuitInterventionEngine',
    'InterventionConfig',
    'MLPInterventionEngine',
    'MLPInterventionConfig',
]
