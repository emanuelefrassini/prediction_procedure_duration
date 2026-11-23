# Research code audit: published protocol vs. repository

The purpose of this audit is **traceability**, not criticism or a claim of invalid published results. The public code appears to contain intermediate/historical experiments; it is not necessarily the exact version used for the publication.

| Subject | Published protocol or result | Observed public code | Recommendation |
|---|---|---|---|
| Input | Manual phase labels from 222 coronary angiography cases, 14 phases, target ETC (seconds) | Large numeric `data/X.csv`, `data/Y.csv` tracked; several loaders assume `(222, 7573, 1)` | Inspect provenance/authorization; do not republish data in test artifacts |
| Model comparison | Six alternatives: InceptionTime, LSTM, LSTM-FCN, Transformer, LSTM-Attention and Ensemble | Matching script families in `models/`, including `models/PAPER_ETC.py` | Keep original experiments; document how to launch each one |
| Repeated evaluation | 100 repeated trials × three folds | `models/PAPER_ETC.py` includes `for trial in range(91,100)` (nine trial indices in this checked-in version) | Recover the exact paper-run configuration before claiming reproduction |
| Ensemble | Mean of **InceptionTime and LSTM-FCN** predictions, after validation-based selection | `models/PAPER_ETC.py` currently computes `0.2 *` each of **five** models | Do not label current ensemble output as the paper's two-model ensemble |
| Validation | Paper describes callbacks on validation loss and evaluation across folds | `models/PAPER_ETC.py` uses `validation_data=(X_test, y_test)` while evaluating the same test fold | Use a disjoint validation subset of training cases in any new experiment |
| Training loss | Paper's methods state SMAPE training loss | `utils/models.py` compiled model definitions inspected use `loss='mae'` | Reconcile actual saved experiment configuration before reproducing the method |
| Metric implementation | Historical SMAPE factor 100 (not 200) | `utils/models.py` defines `100/len(y_true) * sum(abs(error)/(abs(truth)+abs(pred)))` | Guard 0/0 and avoid padding in new computations; document averaging |
| Portability | TensorFlow 2.10.0, Keras, RTX 3090 | Requirements unpinned; Windows `..\data\...` paths, import `from attention import Attention` | Supply lightweight verification separately; optionally apply guarded path/import patch |
| Data products | Patient/procedure-derived workflow annotations | Public repository currently tracks two roughly 43 MB CSV files | Confirm release authorization; do not assume `.gitignore` removes history |

## What is covered by the added tests?

- Correct units and the paper's SMAPE normalization, including the 0/0 corner case.
- Excluding `-1` padded timestamps without treating missing data as correct predictions.
- Whole-procedure deterministic folds and difference between per-case macro and pooled averages.
- Synthetic regression predictions without any patient data or TensorFlow dependency.
- Syntax checks of added files and (when present in the checkout) the principal legacy script.

**What remains unverified:** correctness of all historical TensorFlow training scripts; dataset release permissions; exact 100-trial result reproduction; the clinical error table; training/inference time results; compatibility of legacy custom attention layers with newer Keras. New experiment results should not be labeled 'paper reproduced' until the same preprocessing, selection, architecture, ensemble, splits, loss, seeds and dataset are verified.


## Corrected rerun protocol (new addition)

A separate implementation under [`corrected_protocol/`](../corrected_protocol/)
provides disjoint nested procedure-level splits, a seeded 100-trial default,
masked SMAPE objective, two-model ensemble and test-only final scoring.
See [`corrected-protocol.md`](corrected-protocol.md) for differences,
remaining architecture limitations, and how to run the data-free tests.
**This does not change or retroactively validate** historical
`models/PAPER_ETC.py` or the results published in 2025.
