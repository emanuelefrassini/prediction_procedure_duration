import unittest
import numpy as np
from phase_etc.metrics import (
    mae_seconds, mae_minutes, paper_smape_percent, per_procedure_metrics, horizon_errors_seconds
)


class MetricsTests(unittest.TestCase):
    def test_paper_smape_is_not_twice_normalized(self):
        self.assertAlmostEqual(paper_smape_percent([100], [0]), 100)
        self.assertAlmostEqual(paper_smape_percent([0], [0]), 0)
        self.assertAlmostEqual(paper_smape_percent([100, 50], [50, 50]), 100 / 6)

    def test_padding_does_not_enter_metrics(self):
        y = np.array([60.0, 30.0, 0.0, -1.0, -1.0])
        p = np.array([90.0, 0.0, 0.0, np.nan, 10000.0])
        self.assertAlmostEqual(mae_seconds(y, p), 20)
        self.assertAlmostEqual(mae_minutes(y, p), 1 / 3)

    def test_detect_shape_and_nonfinite_errors(self):
        with self.assertRaises(ValueError):
            mae_seconds([1, 2], [1])
        with self.assertRaises(ValueError):
            mae_seconds([1, 2], [1, float('nan')])
        with self.assertRaises(ValueError):
            mae_seconds([-1, -1], [1, 2])
        with self.assertRaises(ValueError):
            mae_seconds([1], [-2])

    def test_3d_inputs_with_one_feature(self):
        a = np.array([[[60], [0]], [[120], [-1]]], dtype=float)
        b = np.array([[[30], [0]], [[180], [-1]]], dtype=float)
        self.assertAlmostEqual(mae_seconds(a, b), 30)
        r = per_procedure_metrics(a, b)
        self.assertEqual(len(r['per_procedure']), 2)

    def test_macro_and_micro_are_distinct(self):
        truth = np.array([[20, -1, -1], [20, 20, 20]], dtype=float)
        pred = np.array([[10, -1, -1], [20, 20, 20]], dtype=float)
        self.assertAlmostEqual(mae_seconds(truth, pred), 2.5)
        self.assertAlmostEqual(per_procedure_metrics(truth, pred)['macro_mae_seconds'], 5)

    def test_horizon_scoring_uses_tolerance(self):
        t = np.array([[300, 240, -1], [360, 301, -1]], dtype=float)
        p = np.array([[330, 240, -1], [350, 303, -1]], dtype=float)
        x = horizon_errors_seconds(t, p, horizons_minutes=(5,), max_gap_seconds=2)
        self.assertEqual(x[5.0]['n_procedures'], 2)
        self.assertAlmostEqual(x[5.0]['mean_absolute_error_seconds'], 16)
        y = horizon_errors_seconds(t, p, horizons_minutes=(10,), max_gap_seconds=1)
        self.assertIsNone(y[10.0]['mean_absolute_error_seconds'])


if __name__ == '__main__':
    unittest.main()
