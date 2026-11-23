"""Data-free tests for results reporting and project preflight."""
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

from scripts.check_experiment import project_files
from scripts.summarize_results import summarize, _csv_text, main


class ResultSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / 'runs_PAPER_test'
        self.run.mkdir()

    def make_fold(self, trial, interval, fold, model, mae, smape):
        base = self.run / f'trial_{trial}' / f'reduce_dim_{interval}' / f'fold_{fold}' / model
        base.mkdir(parents=True)
        (base / 'mae_per_procedure.txt').write_text(
            f'Average among all the procedures in this fold:  {mae} [min]\n'
            'Procedure number 0 - mae: 1 [min]\n', encoding='utf-8')
        (base / 'SMAPE_per_procedure.txt').write_text(
            f'Average among the procedures in this fold:  {smape} % %\n'
            'Procedure number 0 - SMAPE: 2 % %\n', encoding='utf-8')
        return base

    def test_group_stats_and_std(self):
        self.make_fold(1, 60, 0, 'INCEPTIONTIME', 1.0, 5.0)
        self.make_fold(1, 60, 1, 'INCEPTIONTIME', 2.0, 7.0)
        self.make_fold(2, 60, 0, 'INCEPTIONTIME', 3.0, 9.0)
        self.make_fold(1, 120, 0, 'LSTM', 5.0, 20.0)
        result = summarize(self.run)
        self.assertEqual(len(result['groups']), 2)
        a = result['groups'][0]
        self.assertEqual((a['interval_seconds'], a['model']), (60, 'INCEPTIONTIME'))
        self.assertEqual((a['n_trials'], a['n_completed_folds']), (2, 3))
        self.assertAlmostEqual(a['mean_mae_minutes'], 2.0)
        self.assertAlmostEqual(a['sd_mae_minutes'], 1.0)
        self.assertAlmostEqual(a['mean_smape_percent'], 7.0)
        self.assertAlmostEqual(a['sd_smape_percent'], 2.0)
        self.assertIsNone(result['groups'][1]['sd_mae_minutes'])

    def test_partial_folder_reported_and_strict_mode_fails(self):
        self.make_fold(1, 60, 0, 'LSTM_FCN', 2.0, 8.0)
        partial = self.run / 'trial_1/reduce_dim_60/fold_1/LSTM_FCN'
        partial.mkdir(parents=True)
        (partial / 'mae_per_procedure.txt').write_text('Average among all the procedures in this fold:  4 [min]\n')
        result = summarize(self.run)
        self.assertEqual(result['incomplete'], ['trial_1/reduce_dim_60/fold_1/LSTM_FCN'])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            summarize(self.run, strict=True)

    def test_invalid_scores_rejected(self):
        base = self.make_fold(1, 60, 0, 'Ensemble', 1.0, 6.0)
        (base / 'SMAPE_per_procedure.txt').write_text(
            'Average among the procedures in this fold:  123.5 % %\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Out-of-range'):
            summarize(self.run)

    def test_empty_result_fails(self):
        with self.assertRaisesRegex(ValueError, 'No complete'):
            summarize(self.run)

    def test_csv_and_cli_have_no_procedure_rows(self):
        self.make_fold(1, 60, 0, 'INCEPTIONTIME', 3.5, 5.5)
        summary = summarize(self.run)
        self.assertIn('n_completed_folds', _csv_text(summary))
        output = io.StringIO()
        with redirect_stdout(output):
            main([str(self.run)])
        self.assertIn('3.500', output.getvalue())
        self.assertNotIn('Procedure number 0', output.getvalue())

    def test_preflight_is_file_presence_only(self):
        root = Path(self.temp.name)
        missing = project_files(root)
        self.assertIn('data/X.csv', missing)
        for file in missing:
            path = root / file
            path.parent.mkdir(exist_ok=True, parents=True)
            path.touch()
        self.assertEqual(project_files(root), [])


if __name__ == '__main__':
    unittest.main()
