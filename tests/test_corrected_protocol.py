"""Data-free, TensorFlow-free tests for the corrected training protocol."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from corrected_protocol.protocol import (
    ENSEMBLE_MODELS, MODEL_NAMES, SAMPLING_INTERVALS,
    TARGET_SCALE_SECONDS, fixed_two_model_ensemble, fold_plans,
    masked_smape_numpy, prepare_arrays,
)
from corrected_protocol.run import parse_args


class CorrectedProtocolTests(unittest.TestCase):
    def test_full_trial_schedule_and_disjoint_procedures(self):
        schedule = list(fold_plans(222, trials=100, folds=3, seed=2026))
        self.assertEqual(len(schedule), 300)
        all_case_ids = set(range(222))
        for plan in schedule:
            tr, va, te = map(lambda x: set(x.tolist()),
                             (plan.train, plan.validation, plan.test))
            self.assertTrue(tr and va and te)
            self.assertFalse(tr & va or tr & te or va & te)
            self.assertEqual(tr | va | te, all_case_ids)
        for trial in range(1, 101):
            cases = [p.test.tolist() for p in schedule if p.trial == trial]
            self.assertEqual(len(cases), 3)
            self.assertEqual(sorted(sum(cases, [])), list(range(222)))
        self.assertNotEqual(schedule[0].test.tolist(), schedule[3].test.tolist())

    def test_schedule_reproducible_with_same_seed(self):
        a = list(fold_plans(18, trials=2, seed=47))
        b = list(fold_plans(18, trials=2, seed=47))
        for i, j in zip(a, b):
            np.testing.assert_array_equal(i.train, j.train)
            np.testing.assert_array_equal(i.validation, j.validation)
            np.testing.assert_array_equal(i.test, j.test)

    def test_padding_and_train_only_scaling(self):
        xtrain = np.array([[[2], [4], [-1]], [[4], [6], [-1]]], dtype=float)
        xval = np.array([[[100], [2], [-1]]], dtype=float)
        xtest = np.array([[[-50], [3], [-1]]], dtype=float)
        ytrain = np.array([[[600], [300], [-1]], [[500], [0], [-1]]], dtype=float)
        yval = np.array([[[800], [300], [-1]]], dtype=float)
        ytest = np.array([[[900], [150], [-1]]], dtype=float)
        original = xtrain.copy()
        x, y = prepare_arrays(xtrain, xval, xtest, ytrain, yval, ytest)
        np.testing.assert_array_equal(xtrain, original)
        np.testing.assert_allclose(x[0][0, :2, 0], [1, 1.5])
        self.assertGreater(x[1][0, 0, 0], 20)  # test data never fitted
        self.assertLess(x[2][0, 0, 0], 0)
        self.assertEqual(x[0][0, 2, 0], -1)
        self.assertEqual(y[1][0, 2, 0], -1)
        np.testing.assert_allclose(y[0][0, :2, 0], [1, .5])
        self.assertEqual(TARGET_SCALE_SECONDS, 600.0)

    def test_misaligned_or_interior_padding_rejected(self):
        x = np.array([[[1], [-1], [3]]], dtype=float)
        y = np.array([[[20], [-1], [10]]], dtype=float)
        with self.assertRaisesRegex(ValueError, 'only at the end'):
            prepare_arrays(x, x, x, y, y, y)
        x = np.array([[[1], [2], [-1]]], dtype=float)
        y = np.array([[[20], [-1], [-1]]], dtype=float)
        with self.assertRaisesRegex(ValueError, 'masks do not match'):
            prepare_arrays(x, x, x, y, y, y)

    def test_macro_smape_excludes_padding_and_uses_factor_100(self):
        truth = np.array([[[100], [0], [-1]], [[100], [-1], [-1]]], dtype=float)
        pred = np.array([[[0], [0], [999999]], [[100], [999999], [999999]]], dtype=float)
        self.assertAlmostEqual(masked_smape_numpy(truth, pred), 25.0)
        # Standard factor 200 would have produced 50.0.

    def test_two_model_ensemble_only(self):
        self.assertEqual(ENSEMBLE_MODELS, ('inceptiontime', 'lstm_fcn'))
        pred = {
            'inceptiontime': np.array([[[10.0]], [[20.0]]]),
            'lstm_fcn': np.array([[[30.0]], [[40.0]]]),
            'transformer': np.array([[[999.0]], [[999.0]]]),
        }
        np.testing.assert_allclose(fixed_two_model_ensemble(pred), [[[20]], [[30]]])
        with self.assertRaises(ValueError):
            fixed_two_model_ensemble({'inceptiontime': pred['inceptiontime']})

    def test_end_to_end_fold_calls_fit_without_outer_test_targets(self):
        from unittest.mock import patch
        from corrected_protocol.run import evaluate_fold
        plan = next(fold_plans(12, trials=1, folds=3, seed=91))
        # Every procedure has a distinct target, making train/test leakage
        # observable in the arguments passed to the injected fake trainer.
        x = np.array([[[i], [i + 1], [-1]] for i in range(12)], dtype=float)
        y = np.array([[[100.0 * (i + 1)], [50.0 * (i + 1)], [-1]]
                      for i in range(12)], dtype=float)
        def values(array):
            return {round(float(v), 5) for v in np.ravel(array)}

        allowed_targets = values(100.0 * (plan.train + 1) / TARGET_SCALE_SECONDS)
        allowed_val = values(100.0 * (plan.validation + 1) / TARGET_SCALE_SECONDS)
        prohibited = values(100.0 * (plan.test + 1) / TARGET_SCALE_SECONDS)
        calls = []

        def fake_fit(name, x_train, y_train, x_val, y_val, x_test, *, epochs, seed):
            calls.append(name)
            actual_train = values(y_train[:, 0, 0])
            actual_val = values(y_val[:, 0, 0])
            self.assertEqual(actual_train, allowed_targets)
            self.assertEqual(actual_val, allowed_val)
            self.assertFalse(prohibited & (actual_train | actual_val))
            self.assertEqual(len(x_test), len(plan.test))
            return np.zeros((len(x_test), x_test.shape[1], 1)), {'epochs_run': epochs}

        with patch('corrected_protocol.run.score', return_value={'macro_mae_minutes': 1.0}):
            out = evaluate_fold(x, y, plan, 1, ['inceptiontime', 'lstm_fcn'],
                                epochs=1, seed=99, trainer=fake_fit)
        self.assertEqual(calls, ['inceptiontime', 'lstm_fcn'])
        self.assertEqual(list(out['models']), ['inceptiontime', 'lstm_fcn', 'ensemble'])
        self.assertEqual(out['n_test'], len(plan.test))

    def test_planning_does_not_import_tensorflow_or_read_data(self):
        a = parse_args([])
        self.assertFalse(a.execute)
        self.assertEqual(a.trials, 100)
        self.assertEqual(a.folds, 3)
        self.assertEqual(a.epochs, 1000)
        self.assertEqual(a.intervals, list(SAMPLING_INTERVALS))
        self.assertEqual(a.models, list(MODEL_NAMES))

    def test_training_script_uses_validation_cases_not_test(self):
        src = (Path(__file__).resolve().parents[1] / 'corrected_protocol' / 'run.py').read_text()
        self.assertIn('tf.convert_to_tensor(y_val, dtype=tf.float32)', src)
        self.assertIn('tf.GradientTape()', src)
        self.assertIn('masked_smape_tf(batch_y, prediction)', src)
        fit_src = src.split('def fit_one(', 1)[1].split('def score(', 1)[0]
        self.assertNotIn('y_test', fit_src)
        self.assertIn('fixed_two_model_ensemble(predictions_seconds)', src)


@unittest.skipUnless(importlib.util.find_spec('tensorflow') is not None,
                     'Optional TF loss test; run in Python 3.10 legacy environment')
class TensorFlowLossTests(unittest.TestCase):
    def test_loss_matches_numpy_and_masked_gradients_are_zero(self):
        import tensorflow as tf
        from corrected_protocol.tf_loss import masked_smape_tf
        y = np.array([[[100.], [0.], [-1.]], [[100.], [-1.], [-1.]]], dtype=np.float32)
        prediction = tf.Variable([[[50.], [0.], [500.]], [[100.], [300.], [700.]]])
        with tf.GradientTape() as tape:
            loss = masked_smape_tf(tf.constant(y), prediction)
            total = tf.reduce_mean(loss)
        np.testing.assert_allclose(loss.numpy().mean(), masked_smape_numpy(y, prediction.numpy()), atol=1e-6)
        grad = tape.gradient(total, prediction).numpy()
        np.testing.assert_allclose(grad[y == -1], 0.0, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
