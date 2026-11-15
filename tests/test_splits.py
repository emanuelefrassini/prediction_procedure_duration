import unittest
import numpy as np
from phase_etc.splits import procedure_kfold, procedure_train_val_test_folds


class SplitTests(unittest.TestCase):
    def test_three_fold_disjoint_and_complete(self):
        folds = list(procedure_kfold(222, n_splits=3, seed=42))
        self.assertEqual([len(test) for _, test in folds], [74, 74, 74])
        self.assertEqual(sorted(int(i) for _, test in folds for i in test), list(range(222)))
        for train, test in folds:
            self.assertEqual(len(train), 148)
            self.assertEqual(len(np.intersect1d(train, test)), 0)

    def test_inner_validation_does_not_leak_test(self):
        folds = list(procedure_train_val_test_folds(222, n_splits=3, seed=7))
        self.assertEqual(len(folds), 3)
        for train, validation, test in folds:
            self.assertEqual(set(train) & set(validation), set())
            self.assertEqual(set(train) & set(test), set())
            self.assertEqual(set(validation) & set(test), set())
            self.assertEqual(len(train) + len(validation) + len(test), 222)
            self.assertEqual(len(test), 74)

    def test_invalid_validation_fraction(self):
        with self.assertRaises(ValueError):
            list(procedure_train_val_test_folds(15, validation_fraction=0.0))

    def test_deterministic_per_seed(self):
        a = list(procedure_kfold(17, n_splits=3, seed=123))
        b = list(procedure_kfold(17, n_splits=3, seed=123))
        for (_, x), (_, y) in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_invalid_folds(self):
        for args in ((1, 2), (4, 1), (3, 4)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                list(procedure_kfold(args[0], n_splits=args[1]))


if __name__ == '__main__':
    unittest.main()
