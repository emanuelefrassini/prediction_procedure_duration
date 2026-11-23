import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RepositoryStructureTests(unittest.TestCase):
    def test_citation_and_limitations(self):
        citation = (ROOT / 'CITATION.cff').read_text(encoding='utf-8')
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        audit = (ROOT / 'docs' / 'research-code-audit.md').read_text(encoding='utf-8')
        for text in (citation, readme):
            self.assertIn('10.1080/24699322.2025.2466426', text)
        self.assertIn('not a reproduction', readme.lower())
        self.assertIn('validation', audit.lower())
        self.assertIn('Ensemble', audit)

    def test_new_code_syntax(self):
        paths = list((ROOT / 'phase_etc').glob('*.py')) + list((ROOT / 'scripts').glob('*.py'))
        for path in paths:
            with self.subTest(file=str(path)):
                ast.parse(path.read_text(encoding='utf-8'), filename=str(path))

    def test_legacy_files_if_present(self):
        for name in ('PAPER_ETC.py', 'inceptiontime_ETC.py'):
            file = ROOT / 'models' / name
            if file.exists():
                ast.parse(file.read_text(encoding='utf-8'), filename=str(file))

    def test_caution_for_existing_data(self):
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        self.assertIn('publicly tracked', readme)
        self.assertIn('anonymized', readme)
        gitignore = (ROOT / '.gitignore').read_text(encoding='utf-8')
        self.assertIn('data/*.csv', gitignore)


if __name__ == '__main__':
    unittest.main()
