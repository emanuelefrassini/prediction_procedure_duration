#!/usr/bin/env python3
"""Summarize existing PAPER_ETC fold scores without loading clinical data.

Only the aggregate first line of the model-specific MAE/SMAPE text files is
read. Missing metrics are reported rather than silently counted as complete.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import re
import statistics
import sys

MODELS = ('INCEPTIONTIME', 'LSTM_FCN', 'Ensemble', 'transformer', 'LSTM', 'LSTM_Attention')
AGGREGATE = re.compile(
    r'^Average among (?:all )?the procedures in this fold:\s*'
    r'(?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
)
NUMBERED = re.compile(r'^(trial|reduce_dim|fold)_(\d+)$')
COLUMNS = (
    'interval_seconds', 'model', 'n_completed_folds', 'n_trials',
    'mean_mae_minutes', 'sd_mae_minutes', 'mean_smape_percent', 'sd_smape_percent',
)


def _numbered_directories(folder: Path, prefix: str):
    """Yield matching directories in numeric order; ignore unrelated outputs."""
    if not folder.is_dir():
        return
    found = []
    for p in folder.iterdir():
        match = NUMBERED.fullmatch(p.name)
        if p.is_dir() and match and match.group(1) == prefix:
            found.append((int(match.group(2)), p))
    for item in sorted(found):
        yield item


def _read_metric(path: Path, *, upper_bound: float | None = None):
    with path.open(encoding='utf-8') as f:
        line = f.readline().strip()
    match = AGGREGATE.match(line)
    if not match:
        raise ValueError(f'Expected a fold-level average on first line of {path}')
    score = float(match.group('number'))
    if not math.isfinite(score) or score < 0 or (upper_bound is not None and score > upper_bound):
        raise ValueError(f'Out-of-range metric in {path}: {score}')
    return score


def _summary(values: list[float]):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None


def summarize(run_dir: Path, *, strict: bool = False):
    """Return grouped descriptive statistics of available completed folds."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f'Run directory not found: {run_dir}')
    records: dict[tuple[int, str], list[dict]] = {}
    incomplete = []
    for trial_id, trial_dir in _numbered_directories(run_dir, 'trial'):
        for interval, sampling_dir in _numbered_directories(trial_dir, 'reduce_dim'):
            for fold_id, fold_dir in _numbered_directories(sampling_dir, 'fold'):
                for name in MODELS:
                    model_dir = fold_dir / name
                    if not model_dir.is_dir():
                        continue
                    mae = model_dir / 'mae_per_procedure.txt'
                    smape = model_dir / 'SMAPE_per_procedure.txt'
                    if not mae.is_file() or not smape.is_file():
                        incomplete.append(str(model_dir.relative_to(run_dir)))
                        continue
                    records.setdefault((interval, name), []).append({
                        'trial': trial_id, 'fold': fold_id,
                        'mae': _read_metric(mae),
                        'smape': _read_metric(smape, upper_bound=100.0),
                    })
    if strict and incomplete:
        raise ValueError(f'{len(incomplete)} incomplete model-fold directories found')
    if not records:
        raise ValueError(f'No complete fold-level model scores found in {run_dir}')

    groups = []
    for (interval, model), fold_results in sorted(
            records.items(), key=lambda kv: (kv[0][0], MODELS.index(kv[0][1]))):
        mean_mae, sd_mae = _summary([r['mae'] for r in fold_results])
        mean_smape, sd_smape = _summary([r['smape'] for r in fold_results])
        groups.append({
            'interval_seconds': interval, 'model': model,
            'n_completed_folds': len(fold_results),
            'n_trials': len({r['trial'] for r in fold_results}),
            'mean_mae_minutes': mean_mae, 'sd_mae_minutes': sd_mae,
            'mean_smape_percent': mean_smape, 'sd_smape_percent': sd_smape,
        })
    return {'run_dir': str(run_dir), 'groups': groups, 'incomplete': sorted(incomplete)}


def _csv_text(summary):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=COLUMNS)
    writer.writeheader()
    for row in summary['groups']:
        writer.writerow({key: '' if row[key] is None else row[key] for key in COLUMNS})
    return stream.getvalue()


def _table_text(summary):
    header = f"{'Interval':>8}  {'Model':<16} {'Folds':>5} {'Trials':>6} {'MAE (min)':>12} {'SMAPE (%)':>12}"
    rows = [header, '-' * len(header)]
    for r in summary['groups']:
        rows.append(
            f"{r['interval_seconds']:>7}s  {r['model']:<16} "
            f"{r['n_completed_folds']:>5} {r['n_trials']:>6} "
            f"{r['mean_mae_minutes']:>12.3f} {r['mean_smape_percent']:>12.2f}"
        )
    rows.append(f"Completed model-fold records: {sum(r['n_completed_folds'] for r in summary['groups'])}")
    rows.append(f"Incomplete model folders: {len(summary['incomplete'])}")
    rows.append('These are averages of available fold metrics, not published trial-level estimates.')
    return '\n'.join(rows) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path, help='Existing runs_PAPER_YYYYMMDD-HHMMSS directory')
    parser.add_argument('--format', choices=('table', 'csv', 'json'), default='table')
    parser.add_argument('--output', type=Path, help='Optional output file; stdout is the default')
    parser.add_argument('--strict', action='store_true', help='Fail if any model folder is incomplete')
    args = parser.parse_args(argv)
    try:
        result = summarize(args.run_dir, strict=args.strict)
    except (OSError, ValueError) as exc:
        parser.exit(2, f'error: {exc}\n')
    if args.format == 'csv':
        content = _csv_text(result)
    elif args.format == 'json':
        content = json.dumps(result, indent=2, allow_nan=False) + '\n'
    else:
        content = _table_text(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding='utf-8')
        print(f'Summary saved to {args.output}')
    else:
        sys.stdout.write(content)
    if result['incomplete']:
        print(f'Warning: {len(result["incomplete"])} incomplete model folders excluded.', file=sys.stderr)


if __name__ == '__main__':
    main()
