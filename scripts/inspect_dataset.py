#!/usr/bin/env python3
"""Local CSV shape/padding check. Never prints clinical rows or uploads data."""
import argparse
import csv
from pathlib import Path


def dimensions(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    ncols = None
    nrows = 0
    padded = 0
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = csv.reader(stream)
        for row in rows:
            if not row:
                continue
            if ncols is None:
                ncols = len(row)
            if len(row) != ncols:
                raise ValueError(f"inconsistent number of columns in {path.name} on row {nrows + 1}")
            # Dataset is expected to contain numeric phase/time arrays, not identifiers.
            for value in row:
                number = float(value)
                if number != number or not -float('inf') < number < float('inf'):
                    raise ValueError(f"non-finite numeric entry in {path.name} on row {nrows + 1}")
                padded += number == -1
            nrows += 1
    if nrows == 0:
        raise ValueError(f"empty CSV: {path}")
    return nrows, ncols, padded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    args = parser.parse_args()
    x, y = args.data_dir / 'X.csv', args.data_dir / 'Y.csv'
    rows_x, cols_x, padded_x = dimensions(x)
    rows_y, cols_y, padded_y = dimensions(y)
    if (rows_x, cols_x) != (rows_y, cols_y):
        raise SystemExit('FAIL: X/Y CSV shapes differ')
    print(f'PASS: X.csv and Y.csv shapes match: {rows_x} procedures x {cols_x} timestamps')
    print(f'Padding entries (-1): X={padded_x}, Y={padded_y}')
    if (rows_x, cols_x) != (222, 7573):
        print('NOTE: different from the shape assumed by historical experiment scripts (222, 7573)')
    print('No raw procedure rows, identifiers, or predictions were printed.')


if __name__ == '__main__':
    main()
