"""Deterministic whole-procedure folds; never split timestamps from one case."""
from __future__ import annotations
import numpy as np


def procedure_kfold(n_procedures: int, *, n_splits: int = 3, seed: int = 42):
    """Yield disjoint train/test procedure indices with reproducible shuffling.

    This helper does not create a validation split. Never use the test fold
    for early stopping or hyperparameter selection; carve validation cases
    from each training fold in an actual training experiment.
    """
    if isinstance(n_procedures, bool) or not isinstance(n_procedures, (int, np.integer)):
        raise ValueError("n_procedures must be an integer")
    if n_splits < 2 or n_splits > n_procedures:
        raise ValueError("require 2 <= n_splits <= n_procedures")
    indices = np.random.default_rng(seed).permutation(n_procedures)
    for fold in np.array_split(indices, n_splits):
        test = np.sort(fold)
        train = np.sort(np.setdiff1d(indices, fold))
        yield train, test


def procedure_train_val_test_folds(n_procedures: int, *, n_splits: int = 3, validation_fraction: float = 0.2, seed: int = 42):
    """Yield disjoint (train, validation, test) whole-procedure indices.

    Validation cases are selected strictly from the training side of each
    outer fold. This is an independent protocol helper, NOT a retrofit of
    the published experiment or a guarantee of identical paper splits.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between zero and one")
    for fold_idx, (train_pool, test) in enumerate(procedure_kfold(n_procedures, n_splits=n_splits, seed=seed)):
        if len(train_pool) < 2:
            raise ValueError("too few training procedures to reserve a validation set")
        shuffled = np.random.default_rng(seed + 1009 + fold_idx).permutation(train_pool)
        n_val = max(1, min(len(train_pool) - 1, int(np.ceil(validation_fraction * len(train_pool)))))
        train, val = np.sort(shuffled[n_val:]), np.sort(shuffled[:n_val])
        yield train, val, test
