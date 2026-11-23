#!/usr/bin/env python3
"""Guarded, idempotent documentation/CI fixes for the corrected ETC protocol.

Run from the repository root AFTER extracting the ZIP:
    python scripts/apply_corrected_protocol_docs.py --check
    python scripts/apply_corrected_protocol_docs.py --apply

Only edits known text files. Does not touch data/X.csv, data/Y.csv or Git history.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

README_SECTION = """## Corrected clinical evaluation protocol (separate rerun)

A new [corrected evaluation runner](corrected_protocol/) addresses historical
split leakage, trial counts, ensemble composition, padding-aware SMAPE loss,
and per-model early-stopping state. It uses the **archived architecture factories**
but keeps training, validation and test procedures disjoint.

The default run plan is **100 trials × 3 folds × 5 sampling intervals × 5
architectures**. The command below is a data-free dry run:

```bash
python -m corrected_protocol
```

For a limited real-data training check (requires a separate Python 3.10 /
TensorFlow 2.10 environment), see the [protocol and environment guide](docs/corrected-protocol.md).
The historical `models/PAPER_ETC.py` remains in the repository for provenance;
it is **not** the recommended evaluation runner. The corrected runner has not
been used to reproduce the clinical table in the published article. The
clinical data files are unchanged.

"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text and old not in text:
        return text
    if text.count(old) != 1:
        raise RuntimeError(f'{label}: expected exactly one matching source marker, got {text.count(old)}')
    return text.replace(old, new, 1)


def patch_readme(src: str) -> str:
    old = ("python -m modern_etc --cases 90 --folds 3 --epochs 20 --seed 42 "
           "--width 32 --interval 30 --output /tmp/etc_synthetic.json\n\n\n"
           "See the benchmark documentation and numerical results.\n")
    new = ("python -m modern_etc --cases 90 --folds 3 --epochs 20 --seed 42 "
           "--width 32 --interval 30 --output /tmp/etc_synthetic.json\n````\n\n"
           "See the [benchmark documentation](modern_etc/README.md) and "
           "[numerical results](modern_etc/results/synthetic_example.json).\n")
    # The original issue is a missing closing triple backtick; the temporary
    # four-tick string below gets normalized to three for straightforward tests.
    new = new.replace('````\n', '```\n')
    src = replace_once(src, old, new, 'README modern_etc fence')
    marker = '## Scientific motivation'
    if README_SECTION.strip() not in src:
        src = replace_once(src, marker, README_SECTION + marker, 'README insertion')
    src = src.replace('Existing tracked procedure-derived phase/target data; **review sharing authorization**',
                      'Anonymized procedure-derived phase/target data; original release retained')
    original_governance = (
        'The `data/X.csv` and `data/Y.csv` files are **already tracked in the public Git repository**. '
        'Please verify that redistribution of these procedure-derived data is authorized. '
        'The new `.gitignore` prevents accidental *future* additions of similar files; '
        'it does **not** remove files or their earlier versions from Git history.'
    )
    updated_governance = (
        'The `data/X.csv` and `data/Y.csv` files remain **publicly tracked**. '
        'The maintainer confirms that these procedure-derived data are anonymized and '
        'intended for public distribution; this protocol update does not modify them. '
        'The `.gitignore` prevents accidental *future* additions of similar files; '
        'it does **not** remove files or their earlier versions from Git history.'
    )
    if original_governance in src:
        src = src.replace(original_governance, updated_governance, 1)
    return src


def patch_audit(src: str) -> str:
    note = """
## Corrected rerun protocol (new addition)

A separate implementation under [`corrected_protocol/`](../corrected_protocol/)
provides disjoint nested procedure-level splits, a seeded 100-trial default,
masked SMAPE objective, two-model ensemble and test-only final scoring.
See [`corrected-protocol.md`](corrected-protocol.md) for differences,
remaining architecture limitations, and how to run the data-free tests.
**This does not change or retroactively validate** historical
`models/PAPER_ETC.py` or the results published in 2025.
"""
    if '## Corrected rerun protocol (new addition)' not in src:
        src += '\n' + note
    return src


def patch_repro(src: str) -> str:
    note = """
## Corrected protocol added after the historical audit

See [`docs/corrected-protocol.md`](corrected-protocol.md) and run
`python -m corrected_protocol` to preview the seeded 100 × 3-fold plan
without importing TensorFlow or reading clinical data. Execute only in a
validated Python 3.10 environment, using a limited smoke run first. The
archived training code is retained unchanged apart from a warning comment.
The corrected protocol has **not** been established to be the exact model
configuration used to produce the published results. The maintainer states
that the existing anonymized CSVs were intended for public release; this
patch neither changes them nor rewrites Git history.
"""
    if '## Corrected protocol added after the historical audit' not in src:
        src += '\n' + note
    return src


def patch_legacy(src: str) -> str:
    banner = ("# ARCHIVED EXPERIMENT: This file is preserved for research provenance.\n"
              "# It uses the test fold for Keras validation, nine trials, five-model\n"
              "# ensembling and MAE training. Do not use for corrected evaluation.\n"
              "# See corrected_protocol/ and docs/corrected-protocol.md.\n")
    return src if src.startswith(banner) else banner + src


def patch_ci(src: str) -> str:
    return replace_once(src, '            /phase_etc/\n            /scripts/',
                        '            /phase_etc/\n            /corrected_protocol/\n            /scripts/',
                        'CI sparse checkout')


def changes(root: Path):
    specs = {
        'README.md': patch_readme,
        'docs/research-code-audit.md': patch_audit,
        'docs/reproducibility.md': patch_repro,
        'models/PAPER_ETC.py': patch_legacy,
        '.github/workflows/repository-checks.yml': patch_ci,
    }
    updates = {}
    for rel, fn in specs.items():
        path = root / rel
        old = path.read_text(encoding='utf-8')
        new = fn(old)
        if old != new:
            updates[path] = new
    ignore = root / '.gitignore'
    ignore_old = ignore.read_text(encoding='utf-8')
    ignore_new = ignore_old if '.venv-tf210/' in ignore_old else replace_once(
        ignore_old, '.venv/\n', '.venv/\n.venv-tf210/\n',
        'gitignore legacy environment')
    if ignore_old != ignore_new:
        updates[ignore] = ignore_new
    requirements = root / 'requirements.txt'
    original = requirements.read_text(encoding='utf-8')
    fixed = ('# Reconstructed candidate TF 2.10 environment; Python 3.10 only.\n'
             '# Consult docs/corrected-protocol.md before clinical training.\n'
             '-r requirements-legacy-py310.txt\n')
    if original != fixed:
        if original.strip() != 'numpy\nscikit-learn\ntensorflow\nkeras':
            raise RuntimeError('Unknown requirements.txt content; refusing to overwrite')
        updates[requirements] = fixed
    return updates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    flags = p.add_mutually_exclusive_group(required=True)
    flags.add_argument('--check', action='store_true')
    flags.add_argument('--apply', action='store_true')
    p.add_argument('--repo', type=Path, default=ROOT)
    args = p.parse_args()
    pending = changes(args.repo.resolve())
    if not pending:
        print('No updates needed; documentation and CI fixes already applied.')
        return
    for path in pending:
        print('To update:', path.relative_to(args.repo.resolve()))
    if args.check:
        print('Dry run only; nothing changed.')
        return
    for path, text in pending.items():
        path.write_text(text, encoding='utf-8')
    print('Applied', len(pending), 'text updates; clinical CSVs were not touched.')
    print('Remove already tracked compiled bytecode separately via git rm --cached.')


if __name__ == '__main__':
    main()
