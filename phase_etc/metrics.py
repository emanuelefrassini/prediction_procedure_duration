"""Explicit metrics for remaining-time-to-completion (ETC) predictions.

The paper's historical SMAPE convention is 100 * mean(|prediction - truth| /
(|prediction| + |truth|)). This differs from the often-used 200 *
convention. Zero / zero is defined as zero here. Padding (-1) in ground
truth is omitted; the caller must not score unseen future timestamps.

All temporal values must be in seconds. These utilities require NumPy only.
"""
from __future__ import annotations

import numpy as np


def _normalize_pair(y_true, y_pred):
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"truth/prediction shapes differ: {a.shape} versus {b.shape}")
    if a.ndim not in (1, 2, 3):
        raise ValueError("expected one sequence or a batch of sequences")
    if a.ndim == 3:
        if a.shape[-1] != 1:
            raise ValueError("the trailing feature dimension must equal one")
        a, b = a[..., 0], b[..., 0]
    if a.size == 0:
        raise ValueError("empty input")
    return a, b


def _valid_values(y_true, y_pred, padding_value=-1.0):
    truth, pred = _normalize_pair(y_true, y_pred)
    valid = truth != padding_value
    if not np.any(valid):
        raise ValueError("all timestamps are padded")
    target, estimate = truth[valid], pred[valid]
    if not np.isfinite(target).all() or not np.isfinite(estimate).all():
        raise ValueError("non-finite value at an unpadded timestamp")
    if np.any(target < 0) or np.any(estimate < 0):
        raise ValueError("remaining-time values must be nonnegative; do not silently clip predictions")
    return target, estimate


def mae_seconds(y_true, y_pred, *, padding_value=-1.0):
    """Mean absolute error, in seconds, over all valid timestamps."""
    truth, pred = _valid_values(y_true, y_pred, padding_value)
    return float(np.mean(np.abs(pred - truth)))


def mae_minutes(y_true, y_pred, *, padding_value=-1.0):
    """Mean absolute error, in minutes."""
    return mae_seconds(y_true, y_pred, padding_value=padding_value) / 60.0


def paper_smape_percent(y_true, y_pred, *, padding_value=-1.0):
    """Paper-style SMAPE in percent (0–100), not the 0–200 convention.

    Score is 0 at a timestamp where both ground truth and prediction are 0.
    """
    truth, pred = _valid_values(y_true, y_pred, padding_value)
    numerator = np.abs(pred - truth)
    denominator = np.abs(pred) + np.abs(truth)
    fraction = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return float(100.0 * np.mean(fraction))


def _batched(y_true, y_pred):
    truth, pred = _normalize_pair(y_true, y_pred)
    if truth.ndim != 2:
        raise ValueError("per-procedure analysis expects (procedures, timestamps[, 1])")
    return truth, pred


def per_procedure_metrics(y_true, y_pred, *, padding_value=-1.0):
    """Return per-procedure scores and their macro averages (each case equally weighted).

    This is different from a single pooled timestamp-wise MAE when procedure
    lengths vary. A result is never silently dropped for an empty procedure.
    """
    truth, pred = _batched(y_true, y_pred)
    results = []
    for idx, (t, p) in enumerate(zip(truth, pred)):
        valid = t != padding_value
        results.append({
            "index": idx,
            "valid_timesteps": int(np.sum(valid)),
            "mae_seconds": mae_seconds(t, p, padding_value=padding_value),
            "mae_minutes": mae_minutes(t, p, padding_value=padding_value),
            "smape_percent": paper_smape_percent(t, p, padding_value=padding_value),
        })
    return {
        "per_procedure": results,
        "macro_mae_seconds": float(np.mean([s["mae_seconds"] for s in results])),
        "macro_mae_minutes": float(np.mean([s["mae_minutes"] for s in results])),
        "macro_smape_percent": float(np.mean([s["smape_percent"] for s in results])),
    }


def horizon_errors_seconds(y_true, y_pred, *, horizons_minutes=(5, 10, 15, 20), max_gap_seconds=30.0, padding_value=-1.0):
    """Errors at the observed timestamp nearest each remaining-time horizon.

    This is an explicitly defined illustrative estimator; it is NOT a claim
    to reconstruct the paper's Figure 7 evaluation procedure. A horizon with
    no timestamp within ``max_gap_seconds`` is unavailable (None).
    """
    truth, pred = _batched(y_true, y_pred)
    if max_gap_seconds < 0:
        raise ValueError("max_gap_seconds must be nonnegative")
    out = {}
    for horizon in horizons_minutes:
        if horizon <= 0:
            raise ValueError("horizon minutes must be positive")
        target = float(horizon) * 60.0
        errors = []
        for t, p in zip(truth, pred):
            valid = (t != padding_value) & np.isfinite(t) & np.isfinite(p)
            if not np.any(valid):
                continue
            vt, vp = t[valid], p[valid]
            distances = np.abs(vt - target)
            idx = int(np.argmin(distances))
            if distances[idx] <= max_gap_seconds:
                if vt[idx] < 0 or vp[idx] < 0:
                    raise ValueError("negative remaining time at selected horizon")
                errors.append(float(np.abs(vp[idx] - vt[idx])))
        out[float(horizon)] = {"n_procedures": len(errors), "mean_absolute_error_seconds": float(np.mean(errors)) if errors else None}
    return out
