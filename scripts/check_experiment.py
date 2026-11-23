#!/usr/bin/env python3
"""Preflight checks for the paper experiment; never imports models or loads CSVs."""
from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    'models/PAPER_ETC.py', 'utils/models.py', 'utils/paper_training.py',
    'data/X.csv', 'data/Y.csv', 'CITATION.cff',
)


def project_files(root: Path):
    """Return missing required project files, without opening the CSVs."""
    return [name for name in REQUIRED if not (root / name).is_file()]


def installed_version(package: str):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Repository root')
    parser.add_argument('--skip-data', action='store_true',
                        help='Skip required CSV presence checks (for CI sparse checkout)')
    parser.add_argument('--training', action='store_true',
                        help='Require Python 3.10 and TensorFlow 2.10 installation')
    args = parser.parse_args(argv)
    print(f'Python: {platform.python_version()}')
    missing = project_files(args.root)
    if args.skip_data:
        missing = [name for name in missing if not name.startswith('data/')]
    checked_count = len(REQUIRED) - (2 if args.skip_data else 0)
    print(f'Project files: {checked_count - len(missing)}/{checked_count} present')
    if missing:
        for name in missing:
            print(f'  MISSING {name}')
    for filename in (() if args.skip_data else ('data/X.csv', 'data/Y.csv')):
        path = args.root / filename
        if path.is_file():
            print(f'{filename}: {path.stat().st_size / (1024 ** 2):.1f} MiB')
    print('Dataset: skipped by request' if args.skip_data else 'Dataset contents: not read')

    tf_version = installed_version('tensorflow')
    print(f'TensorFlow package: {tf_version or "not installed"}')
    valid_training = (sys.version_info[:2] == (3, 10) and tf_version is not None
                      and tf_version.split('.')[:2] == ['2', '10'])
    if args.training:
        print('TensorFlow training environment:', 'OK' if valid_training else 'INCOMPATIBLE')
    print('Ready for repository checks.' if not missing else 'Repository files are incomplete.')
    if missing or (args.training and not valid_training):
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
