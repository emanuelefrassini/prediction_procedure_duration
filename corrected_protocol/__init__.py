"""Independent, auditable evaluation protocol for the archived ETC models.

This is not a claim to reconstruct the precise 2025 paper-run configuration.
"""
from .protocol import (
    ENSEMBLE_MODELS, MODEL_NAMES, SAMPLING_INTERVALS,
    FoldPlan, fold_plans, fixed_two_model_ensemble,
    prepare_arrays, masked_smape_numpy,
)

__all__ = [
    'ENSEMBLE_MODELS', 'MODEL_NAMES', 'SAMPLING_INTERVALS',
    'FoldPlan', 'fold_plans', 'fixed_two_model_ensemble',
    'prepare_arrays', 'masked_smape_numpy',
]
