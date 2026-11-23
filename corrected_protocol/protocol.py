"""Pure-NumPy checks, target handling and complete nested fold planning.

No patient data or TensorFlow is imported at module import time. Whole
procedures remain grouped across the training/validation/test partitions.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from phase_etc.splits import procedure_train_val_test_folds

SAMPLING_INTERVALS = (5, 10, 30, 60, 120)
MODEL_NAMES = ('lstm', 'lstm_attention', 'lstm_fcn', 'transformer', 'inceptiontime')
ENSEMBLE_MODELS = ('inceptiontime', 'lstm_fcn')
TARGET_SCALE_SECONDS = 600.0   # constant chosen in advance, never fitted
PADDING_VALUE = -1.0


@dataclass(frozen=True)
class FoldPlan:
    trial: int
    fold: int
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def fold_plans(n_cases: int, *, trials: int = 100, folds: int = 3,
               seed: int = 2026, validation_fraction: float = 0.2):
    """Yield 100 seeded repetitions of three disjoint *procedure-level* folds.

    Uses one independent inner validation sample from each outer training
    pool. No outer test case participates in fitting or epoch selection.
    """
    if isinstance(trials, bool) or not isinstance(trials, (int, np.integer)) or trials < 1:
        raise ValueError('trials must be a positive integer')
    if isinstance(folds, bool) or not isinstance(folds, (int, np.integer)) or folds < 2:
        raise ValueError('folds must be an integer >= 2')
    if not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    for trial in range(1, int(trials) + 1):
        outer_seed = int(seed) + trial - 1
        for fold, (train, val, test) in enumerate(procedure_train_val_test_folds(
                n_cases, n_splits=int(folds), validation_fraction=validation_fraction,
                seed=outer_seed)):
            yield FoldPlan(trial, fold, train, val, test)


def _as_pair(x: np.ndarray, y: np.ndarray):
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 3 or y.ndim != 3 or x.shape != y.shape or x.shape[-1] != 1:
        raise ValueError('expected aligned (procedures, timestamps, 1) arrays')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('nonfinite sequence values')
    valid_x = x[..., 0] != PADDING_VALUE
    valid_y = y[..., 0] != PADDING_VALUE
    if not np.array_equal(valid_x, valid_y):
        raise ValueError('phase/target padding masks do not match')
    if not valid_y.any(axis=1).all():
        raise ValueError('procedures without valid target timestamps')
    # Padding must be a suffix, never an internal gap followed by valid data.
    if np.any(valid_y[:, 1:] & ~valid_y[:, :-1]):
        raise ValueError('padding must occur only at the end of each procedure')
    if np.any(y[..., 0][valid_y] < 0):
        raise ValueError('unmasked remaining times must be >= 0 seconds')
    return x, y


def prepare_arrays(x_train, x_val, x_test, y_train, y_val, y_test):
    """Train-only phase scaling; fixed target scale; retain -1 padding exactly.

    The source is never mutated. No scaler is fitted on validation or test
    cases. The SMAPE objective is invariant under positive target scaling.
    """
    pairs = [_as_pair(x, y) for x, y in zip(
        (x_train, x_val, x_test), (y_train, y_val, y_test))]
    (xt, yt), (xv, yv), (xe, ye) = pairs
    training_values = xt[xt != PADDING_VALUE]
    low, high = float(training_values.min()), float(training_values.max())
    width = high - low
    if width <= 0:
        width = 1.0

    def transform_x(x):
        # Keep valid inputs strictly positive: an archived LSTM-FCN branch
        # uses Keras Masking() with default mask_value=0, while other branches
        # use -1. This avoids treating a genuine minimum phase as padding.
        return np.where(x == PADDING_VALUE, PADDING_VALUE,
                        1.0 + (x - low) / width).astype(np.float32)

    def transform_y(y):
        return np.where(y == PADDING_VALUE, PADDING_VALUE,
                        y / TARGET_SCALE_SECONDS).astype(np.float32)

    return ((transform_x(xt), transform_x(xv), transform_x(xe)),
            (transform_y(yt), transform_y(yv), transform_y(ye)))


def fixed_two_model_ensemble(predictions: dict[str, np.ndarray]):
    """Fixed arithmetic mean on a common seconds scale, with no test tuning."""
    if not all(k in predictions for k in ENSEMBLE_MODELS):
        raise ValueError('InceptionTime and LSTM-FCN predictions both required')
    a, b = (np.asarray(predictions[k], dtype=np.float64) for k in ENSEMBLE_MODELS)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('ensemble predictions must match and be finite')
    return (a + b) / 2.0


def masked_smape_numpy(y_true, y_pred):
    """Unweighted macro-of-procedure SMAPE (0..100), excluding -1 suffixes.

    This is the same explicitly defined objective as masked_smape_tf, and a
    diagnostic helper for unit tests. 0/0 contributes zero.
    """
    t, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if t.shape != p.shape or t.ndim != 3 or t.shape[-1] != 1:
        raise ValueError('expected paired rank-three sequence arrays')
    valid = t != PADDING_VALUE
    if not valid.any(axis=(1, 2)).all() or not np.isfinite(t[valid]).all() or not np.isfinite(p[valid]).all():
        raise ValueError('missing valid targets or nonfinite values')
    tt, pp = np.where(valid, t, 0.0), np.where(valid, p, 0.0)
    ratio = np.divide(np.abs(tt - pp), np.abs(tt) + np.abs(pp),
                      out=np.zeros_like(tt), where=(np.abs(tt) + np.abs(pp)) > 0)
    return float(np.mean(100.0 * ratio.sum(axis=(1, 2)) / valid.sum(axis=(1, 2))))
