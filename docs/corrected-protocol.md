# Corrected evaluation protocol — archived ETC model architectures

This is a **new, explicitly versioned training protocol** addressing problems
identified in the historical `models/PAPER_ETC.py`. It is *not* a verified
reproduction of the experiments reported in the 2025 paper. The original script
remains available for provenance; it still contains the historical settings.

The repository's anonymized phase/target CSVs are preserved, as requested.
No clinical records or model weights are bundled with this extension.

## Changes relative to the checked-in historical runner

| Aspect | Historical `models/PAPER_ETC.py` | New `corrected_protocol/` |
|---|---|---|
| Outer test data | Used as Keras validation data | **Never** passed to `fit` or used for epoch selection |
| Validation | Same cases as test | New independent procedure-level subset of outer training fold |
| Trials | `range(91,100)` (9 iterations) | Seeded trials numbered 1–100 by default |
| Ensemble | Equal mean of 5 models | Fixed equal mean of **InceptionTime and LSTM-FCN only** |
| Training objective | MAE in archived factories | Masked **per-procedure SMAPE** (factor 100) |
| Target scaling | MinMax fitted with padded `-1` | Fixed scale of 600 seconds, preserving `-1` and no fitting |
| Phase scaling | MinMax over all training entries including padding | Valid phase inputs only; fit on training cases, valid inputs start at 1 and padding stays `-1` |
| Early stopping / LR scheduling | One shared callback variable reused | Fresh independent state in every model fit |
| Test metrics | Mixed padding/inverse scaling handling | Real units (seconds), padding excluded; macro per-procedure |
| Dataset access | Module import launches expensive training | Explicit `--execute` flag; default dry run |
| Repeated splits | Unseeded `KFold` | Deterministic seed recorded per trial |
| Training batch order | Keras `fit(..., shuffle=False)`; paper states batches were not shuffled | No per-epoch permutation of training procedures (paper-aligned) |

The historical factory networks are reused **without claiming architectural
identity with the trained models in the publication**. The existing
LSTM-FCN, InceptionTime and Transformer definitions may use future sequence
positions (noncausal convolutions/attention). This protocol corrects *fold
isolation and evaluation*, not the real-time causality of those models.
The separate `modern_etc` synthetic benchmark performs explicit prefix
causality tests, but differs architecturally from the historical research.

## Environment

The candidate legacy environment is in `requirements-legacy-py310.txt`.
Create a new **Python 3.10** virtual environment for TensorFlow 2.10:

```bash
python3.10 -m venv .venv-tf210
source .venv-tf210/bin/activate
python -m pip install -r requirements-legacy-py310.txt
```

These versions are **compatibility pins, not recovered historical package
versions or a fully hashed lockfile**. The TensorFlow stack has not been
installed or trained in the data-free project CI. Before presenting new
clinical scores, the exact dependency combination must be installed and the
runner tested in that environment. Save `python -m pip freeze`, environment
and hardware information with any eventual experiment outputs.

## Verify without reading the clinical dataset

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m corrected_protocol
```

The default dry run displays the full plan: **100 trials × 3 folds × 5
sampling intervals × 5 models = 7,500 model fits**. This is resource-intensive.
For an initial real-data smoke experiment, run only after your Python 3.10
TensorFlow environment is working:

```bash
python -m corrected_protocol --execute --trials 1 --epochs 2 \
  --intervals 60 --models inceptiontime lstm_fcn
```

For the complete corrected protocol:

```bash
python -m corrected_protocol --execute
```

Raw input is read from `data/X.csv`, `data/Y.csv` by default; use
`--data-dir /path/to/data` to override. Only aggregate model metrics, run
settings, and train/validation/test sizes are written. The output defaults
to `models/runs/corrected_summary.json` and is ignored by Git.

## Training and evaluation details

For each trial, all 222 procedures are partitioned into three outer test
folds. A fixed fraction (default 20%) of the *remaining* procedures is used
for early stopping and LR scheduling. The new runner uses an explicit
`tf.GradientTape` loop rather than relying on automatic Keras output-mask
weighting; padded targets are excluded in the differentiable objective.
The corrected loop preserves procedure order within each training fold across
all epochs. This follows the paper's non-shuffled batching description;
there is no claim that its original run order or random seeds are recovered.
The test cases are evaluated only after
each model is fitted. Each trial uses a recorded seed; each fitted model has
its own deterministic seed. Different trials produce different outer folds.

The masked loss implements per-case

\[
\operatorname{SMAPE}_{100}(y,\hat y)
=\frac{100}{|V|}\sum_{t\in V}
 \frac{|\hat y_t-y_t|}{|\hat y_t|+|y_t|},
\]

where \(V\) comprises valid (non-`-1`) timestamps, and `0/0` is zero. This
returns one value per case before Keras averages the batch, avoiding a
long-procedure bias within each batch. Evaluation uses the same SMAPE factor
and computes equal-weight **macro means over procedures**. Published values
must not be replaced until the aggregation and architecture match the original
study and the entire experiment has actually been executed.

No hyperparameter search is done on held-out test results. The two-model
ensemble is fixed in advance, with equal arithmetic weights, without
selecting the component models from performance on test folds.

## Remaining limitations

- The legacy architecture factories may have version-specific Keras behaviour;
  **real clinical training is not performed as part of this documentation-only update; the user has already completed limited one-epoch smoke runs**.
- This protocol makes an explicit interpretation of the paper's SMAPE loss and
  the training/validation split. Neither is proven to be the exact historical
  configuration used to create the published table.
- Repeating cross-validation on one dataset does not provide external clinical
  validation. Results from repeated trials are correlated.
- Numerical phase labels are treated as scalar inputs by the historical
  architectures. Phase inputs are offset away from zero to avoid the
  historical LSTM-FCN branch's default `Masking(mask_value=0)` incorrectly
  dropping a valid minimum phase; its handling of padded suffixes remains
  architecturally imperfect. A categorical embedding could be preferable, but changing
  it would be a separate model-design experiment.
- Keras callback behaviour, early-stopping epoch selection and architecture
  causality must be checked empirically in the TensorFlow 2.10 environment.
- **The current publication table in the root README remains a citation from
  the paper, not a rerun.**
