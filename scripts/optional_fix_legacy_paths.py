#!/usr/bin/env python3
"""Optional guarded portability patches; does not change model semantics.

Historical source is preserved by default. --check shows proposed changes,
--apply creates backups and updates files only where all known patterns match.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil


def proposed_changes(repo: Path):
    runner = repo / 'models' / 'PAPER_ETC.py'
    helper = repo / 'utils' / 'models.py'
    if not runner.is_file() or not helper.is_file():
        raise FileNotFoundError('Expected models/PAPER_ETC.py and utils/models.py at repository root')
    replacements = {
        runner: [
            ('def load_data(file_path):', 'def load_data(file_path=None):'),
            ('X_path = os.path.join(current_dir, "..\\data\\X.csv")',
             'X_path = os.path.join(os.environ.get("ETC_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")), "X.csv")'),
            ('Y_path = os.path.join(current_dir, "..\\data\\Y.csv")',
             'Y_path = os.path.join(os.environ.get("ETC_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")), "Y.csv")'),
        ],
        helper: [('from attention import Attention', 'from utils.attention_layer import Attention')],
    }
    updates = {}
    for path, changes in replacements.items():
        text = path.read_text(encoding='utf-8')
        changed = []
        for old, new in changes:
            if old in text:
                if text.count(old) != 1:
                    raise ValueError(f'Expected exactly one occurrence in {path.name}: {old}')
                text = text.replace(old, new, 1)
                changed.append(old)
            elif new not in text:
                raise ValueError(f'Unexpected source layout in {path}: could not locate {old!r}')
        if changed:
            updates[path] = (text, changed)
    return updates


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--check', action='store_true', help='Preview changes only')
    g.add_argument('--apply', action='store_true', help='Apply guarded patch, with backups')
    ap.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1], help='Repository root')
    args = ap.parse_args()
    changes = proposed_changes(args.repo)
    if not changes:
        print('Already patched: no changes necessary.')
        return
    for path, (_, patterns) in changes.items():
        print(f'{path.relative_to(args.repo)}: {len(patterns)} proposed replacement(s)')
    if args.check:
        print('Dry run only; no files modified.')
        return
    # Confirm all patterns before writing anything.
    for path, (new_text, _) in changes.items():
        backup = path.with_suffix(path.suffix + '.bak')
        if backup.exists():
            raise FileExistsError(f'Backup already exists: {backup}; refusing to overwrite')
        shutil.copy2(path, backup)
        path.write_text(new_text, encoding='utf-8')
        print(f'Updated {path.relative_to(args.repo)}; backup: {backup.name}')
    print('Compatibility patch applied. Training/evaluation logic not changed.')


if __name__ == '__main__':
    main()
