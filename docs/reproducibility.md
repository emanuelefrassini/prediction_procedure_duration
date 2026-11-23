# Reproducibility guide

This project accompanies the 2025 peer-reviewed benchmark (DOI: [10.1080/24699322.2025.2466426](https://doi.org/10.1080/24699322.2025.2466426)). **Published metrics are cited from the paper and have not been independently reproduced by the lightweight checks.**

## Scientific setup reported in the paper

- Cohort: 222 diagnostic coronary angiography procedures at one hospital, annotated with 14 clinical workflow phases.
- Target: remaining time until procedure completion, in **seconds**.
- Input: phase labels obtained from manual review of video recordings; raw video is *not* a required network input for the models in this benchmark.
- Temporal sampling intervals: **5, 10, 30, 60 and 120 seconds**. The paper describes **100 repeated experiments with three-fold cross-validation**.
- Compared architectures: InceptionTime, LSTM, LSTM-FCN, Transformer, LSTM with attention, and an ensemble of InceptionTime and LSTM-FCN.
- Historical software: TensorFlow **2.10.0**, Keras, and NVIDIA RTX 3090 GPU (as reported in the paper).
- Training and evaluation need patient/procedure-level partitioning and a **separate validation partition** for model choice and early stopping. Never select hyperparameters with the held-out test fold.

## Quick checks requiring no clinical dataset or TensorFlow

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-validation.txt
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/smoke_etc.py
```

These verify metric definitions, masking, repeatable procedure-level fold allocation (including a **disjoint inner validation split**) and a synthetic regression example. **They do not train or evaluate any deep-learning model.**

## Checking the existing numeric CSVs locally

```bash
python scripts/inspect_dataset.py --data-dir data
```

The script verifies matching dimensions, numeric values and padding counts without printing procedure rows or identifiers. The historical runner expects `(222, 7573)` for each numeric CSV and reshapes to `(222, 7573, 1)`. With a sampling interval of 60 seconds, selecting `[:, ::60, :]` leaves 127 time points from a sequence of 7573. The raw clinical phase samples in the repository are **already tracked**, and you should independently verify whether their public availability is authorized.

### Running the historical TensorFlow experiments

The `models/` scripts are archived research code rather than a plug-and-play, pinned reproduction environment. The historical scripts contain hard-coded paths and version-sensitive TensorFlow/Keras APIs. The published article cites TF 2.10; its installation is environment-specific and does not generally work under the newest Python versions. Keep this environment separate from the data-free tests above. Use approved data only.

A limited path/import patch is available without changing the training logic:

```bash
python scripts/optional_fix_legacy_paths.py --check
# inspect source and decide whether to opt in:
python scripts/optional_fix_legacy_paths.py --apply
```

If applied, set `ETC_DATA_DIR=/absolute/path/to/approved/data` or use the repository `data/` directory. Backups (`*.bak`) are saved and ignored. This optional patch **does not** fix the methodological mismatches described in [research-code-audit.md](research-code-audit.md), and **does not** guarantee that the historical scripts run under a modern Keras version. Do not launch a 100-trial, three-fold training campaign until reviewing and resolving the listed differences.

## Metric conventions

In the paper and original `utils/models.py`, the SMAPE function is

`100 × mean(|prediction − truth| / (|truth| + |prediction|))`

which ranges from 0 to 100%, not the sometimes-used 0 to 200% version. In the added `phase_etc.metrics`, `0/0` contributes zero; timestamps with true target `-1` are excluded as padding. Per-procedure **macro** averages weigh each procedure equally, while pooled timestamp averages weigh longer procedures more. Specify which aggregation is used in any new experiment.

`horizon_errors_seconds` offers an explicitly defined nearest-observed-timestamp analysis at 5/10/15/20 min remaining. It is a validation helper, not a verified reconstruction of the paper's Figure 7 methodology.

## Responsible sharing

The paper's open-access **CC BY 4.0** publication license does not automatically grant a software or clinical-data redistribution license. No code license is asserted in this update. Never add recordings, patient identifiers, weights, experiment output or intermediate clinical feature arrays to a public repo without approval. `.gitignore` only prevents **new untracked additions**; it cannot remove or de-publish the `data/X.csv` and `data/Y.csv` files that are already in Git history.


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
