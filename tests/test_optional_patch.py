import tempfile
import unittest
from pathlib import Path
from scripts.optional_fix_legacy_paths import proposed_changes


class OptionalPatchTests(unittest.TestCase):
    def test_patterns_are_guarded_and_reentrant(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / 'models').mkdir()
            (repo / 'utils').mkdir()
            p = repo / 'models' / 'PAPER_ETC.py'
            p.write_text('def load_data(file_path):\n    X_path = os.path.join(current_dir, "..\\data\\X.csv")\n    Y_path = os.path.join(current_dir, "..\\data\\Y.csv")\n')
            q = repo / 'utils' / 'models.py'
            q.write_text('from attention import Attention\n')
            updates = proposed_changes(repo)
            self.assertEqual(len(updates), 2)
            for path, (source, _) in updates.items():
                path.write_text(source)
            self.assertEqual(proposed_changes(repo), {})


if __name__ == '__main__':
    unittest.main()
