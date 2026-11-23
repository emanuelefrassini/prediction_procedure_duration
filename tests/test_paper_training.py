"""Lightweight checks of the experiment's procedure-level data handling."""
import ast
from pathlib import Path
import unittest
import numpy as np

from utils.paper_training import (
    PADDING, TARGET_SCALE_SECONDS, TargetScale,
    prepare_fold_arrays, smape, split_train_validation,
)

ROOT = Path(__file__).resolve().parents[1]


class PaperTrainingTests(unittest.TestCase):
    def test_inner_validation_is_disjoint_and_deterministic(self):
        outer_train = np.arange(148)
        a, b = split_train_validation(outer_train, seed=73)
        c, d = split_train_validation(outer_train, seed=73)
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
        self.assertEqual(len(a), 118)
        self.assertEqual(len(b), 30)
        self.assertEqual(set(a) & set(b), set())
        self.assertEqual(set(a) | set(b), set(outer_train))

    def test_padding_and_train_only_scaling(self):
        x = np.array([[[2], [3], [-1]], [[3], [4], [-1]],
                      [[4], [5], [-1]], [[100], [101], [-1]]], dtype=float)
        y = np.array([[[120], [60], [-1]], [[100], [50], [-1]],
                      [[90], [30], [-1]], [[200], [50], [-1]]], dtype=float)
        xtr, xv, xe, ytr, yv, ye = prepare_fold_arrays(
            x, y, [0, 1], [2], [3])
        self.assertEqual(float(xtr[0, 0, 0]), 1)
        self.assertGreater(float(xe[0, 0, 0]), 2)
        self.assertEqual(float(xtr[0, -1, 0]), PADDING)
        self.assertEqual(float(ye[0, -1, 0]), PADDING)
        np.testing.assert_allclose(TargetScale().inverse_transform(ytr)[:, :-1, :], y[:2, :-1, :])
        self.assertEqual(TARGET_SCALE_SECONDS, 600.0)

    def test_invalid_padding_rejected(self):
        x = np.array([[[1], [-1], [2]], [[2], [3], [-1]], [[3], [4], [-1]]])
        y = x.copy()
        with self.assertRaises(ValueError):
            prepare_fold_arrays(x, y, [0], [1], [2])

    def test_smape_factor_and_zero_division(self):
        self.assertAlmostEqual(smape([1, 0, 0, -1], [0, 0, 0, 999]), 100/3)
        self.assertAlmostEqual(smape([10, 10], [5, 10]), 100/6)

    def test_primary_script_is_paper_aligned(self):
        runner = (ROOT / 'models' / 'PAPER_ETC.py').read_text(encoding='utf-8')
        ast.parse(runner)
        self.assertIn('ETC_TRIALS', runner)
        self.assertIn('split_train_validation(', runner)
        self.assertIn('prepare_fold_arrays(', runner)
        self.assertIn('0.5 * (y_test_predict_INCEPTIONTIME + y_test_predict_LSTM_FCN)', runner)
        self.assertEqual(runner.count('= train_model('), 5)
        self.assertIn('loss=masked_smape_tf', runner)
        self.assertNotIn('validation_data=(X_test, y_test)', runner)


if __name__ == '__main__':
    unittest.main()
