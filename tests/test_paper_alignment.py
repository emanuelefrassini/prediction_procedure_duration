"""Data-free regression checks of the published batch-order protocol."""
from __future__ import annotations

import inspect
import unittest

import numpy as np
from corrected_protocol.run import fit_one, unshuffled_procedure_batches


class PaperAlignmentTests(unittest.TestCase):
    def test_nonshuffled_batches_cover_every_training_procedure(self):
        batches = list(unshuffled_procedure_batches(11, 4))
        self.assertEqual([b.tolist() for b in batches],
                         [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]])
        np.testing.assert_array_equal(np.concatenate(batches), np.arange(11))

    def test_nonshuffled_batch_edge_cases(self):
        self.assertEqual([a.tolist() for a in unshuffled_procedure_batches(3, 8)],
                         [[0, 1, 2]])
        for n, size in ((0, 1), (3, 0), (3, -1)):
            with self.subTest(n=n, size=size), self.assertRaises(ValueError):
                list(unshuffled_procedure_batches(n, size))

    def test_training_loop_uses_unshuffled_batches(self):
        source = inspect.getsource(fit_one)
        self.assertIn('for idx in unshuffled_procedure_batches(', source)
        self.assertNotIn('rng.permutation(', source)


if __name__ == '__main__':
    unittest.main()
