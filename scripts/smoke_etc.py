#!/usr/bin/env python3
"""Data-free smoke test for deterministic phase-based ETC metric utilities."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase_etc import mae_minutes, paper_smape_percent, per_procedure_metrics, horizon_errors_seconds, procedure_kfold


def main():
    rng = np.random.default_rng(2025)
    n, length = 12, 35
    truth = np.full((n, length), -1.0)
    prediction = np.full((n, length), -1.0)
    # Synthetic 60-second samples, varied lengths, no real procedures or patient data.
    for idx in range(n):
        steps = 22 + idx
        remaining = np.arange(steps - 1, -1, -1, dtype=float) * 60.0
        truth[idx, :steps] = remaining
        prediction[idx, :steps] = np.maximum(0, remaining + rng.normal(0.0, 25.0, steps))

    results = per_procedure_metrics(truth, prediction)
    assert len(results['per_procedure']) == n
    assert 0 <= results['macro_smape_percent'] <= 100
    assert 0 < results['macro_mae_minutes'] < 2
    assert np.isclose(mae_minutes(truth, prediction), np.mean(np.abs(prediction[truth != -1] - truth[truth != -1])) / 60.0)
    assert paper_smape_percent(np.array([0.0]), np.array([0.0])) == 0

    folds = list(procedure_kfold(n, seed=2025))
    seen = [int(i) for _, test in folds for i in test]
    assert sorted(seen) == list(range(n))
    for train, test in folds:
        assert not np.intersect1d(train, test).size
    horizon = horizon_errors_seconds(truth, prediction, horizons_minutes=(5, 10, 15, 20), max_gap_seconds=0)
    assert all(row['n_procedures'] == n for row in horizon.values())
    print(f"PASS: {n} synthetic procedures, 3 disjoint deterministic folds")
    print(f"PASS: macro MAE = {results['macro_mae_minutes']:.3f} min, paper-style SMAPE = {results['macro_smape_percent']:.2f}%")
    print("PASS: -1 padding exclusion, zero-denominator SMAPE, horizon scoring")
    print("Synthetic ETC smoke test PASSED (not a reproduction of the clinical benchmark).")


if __name__ == '__main__':
    main()
