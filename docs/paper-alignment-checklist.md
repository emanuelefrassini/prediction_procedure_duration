# Paper-alignment checklist and limits of verification

This document distinguishes **a corrected, paper-aligned implementation** from
**a verified reproduction of published numerical results**. It relates to
Frassini et al., *Computer Assisted Surgery* (2025), DOI
[10.1080/24699322.2025.2466426](https://doi.org/10.1080/24699322.2025.2466426).

## Which code to show a reviewer

- [`../corrected_protocol/`](../corrected_protocol/): the recommended
  TensorFlow 2.10 training/evaluation protocol. It builds the full model
  architectures from the archived `utils/models.py`; it does **not** train the
  smaller PyTorch models in `modern_etc`.
- `models/PAPER_ETC.py`: historical source kept as evidence of the archived
  state. It still uses the outer test fold for Keras validation, averages five
  models, and runs only trial indices 91–99. **Do not use it to demonstrate a
  leakage-free paper reproduction.**
- `modern_etc/`: an independent **reduced** causal PyTorch **synthetic-data**
  benchmark. Its results are not clinical results and are not a reproduction.

## Original audit, with present disposition

| Audit finding | Disposition / evidence | Limit |
| --- | --- | --- |
| Public `data/X.csv`, `data/Y.csv` | Retained unchanged; maintainer confirms the phase/target data are anonymized and intended to be public | This is a maintainer statement, not an independent data-governance certification |
| Test fold reused for early stopping | `corrected_protocol/run.py` creates an inner procedure-level validation split and supplies only training and validation cases to `fit_one` | Historical `PAPER_ETC.py` remains uncorrected for provenance; the publication's actual experiment code has not been recovered |
| Five-model rather than two-model ensemble | Corrected runner averages only InceptionTime and LSTM-FCN test predictions, with equal weights | The two-model choice is fixed from the reported method, not re-selected per trial |
| Nine rather than 100 trials | Default configuration plans 100 seeded trials, each with 3 outer folds and all five sampling intervals | Full run has not been executed, and original fold seeds are unknown |
| MAE rather than SMAPE training loss | Explicit differentiable, masked SMAPE loss, factor 100 and `0/0 = 0` | Per-procedure macro loss and target scale are specified implementation choices, not independently verified historical settings |
| 100 versus 200 SMAPE convention | 100-factor convention retained in loss, evaluation, and docs | Do not compare the values with 200-factor SMAPE without conversion |
| Reproducibility hygiene | Candidate TF 2.10 / Python 3.10 pins, WSL-compatible paths, `.gitignore` for bytecode; tracked bytecode removed in cleanup commit | Candidate pins are compatibility versions, not a recovered exact historical lockfile |
| Training batch shuffling | Corrected runner now keeps procedure order within each epoch, as described by the paper | Exact historical random ordering is unknown |

## Architectural fidelity and unresolved scientific limitation

The corrected runner invokes the existing full-size archived model factories,
with their original hyperparameter choices. Its greater fidelity comes from
**preserving those architectures**, not from changing them to smaller or
strictly causal substitutes. In particular, the archived InceptionTime and
LSTM-FCN models include operations that can depend on future sequence
positions (`same` convolutions; in LSTM-FCN, global sequence pooling).
These operations mean that a prediction for an earlier timestamp is not
necessarily a *real-time, prefix-only* prediction. This potential architectural
leakage is **distinct** from outer-fold test leakage. Altering the networks to
be causal would be a new architecture and must be reported as such; it is
not a minimal correction to the original paper-aligned implementation.

The corrected training pipeline also makes explicit choices where the paper
and archived code do not establish a unique implementation, including the
20% inner-validation fraction, seed schedule, valid-only phase scaling, fixed
600-second target scale, macro-of-procedures masked SMAPE and custom
`GradientTape` optimization. These are **documented deviations**, not claims
of recovering the original experimental configuration.

## Verification already available

- Data-free unit tests check disjoint folds, trial scheduling, target masking,
  ensemble definition and absence of test targets from model fitting.
- In the maintainer's Python 3.10.22 / TensorFlow 2.10.0 environment, the
  TensorFlow masked-loss integration test passed.
- One-epoch, three-fold real-data smoke runs completed for LSTM, InceptionTime,
  and LSTM-FCN. Their results only show that the runner can execute; they do
  **not** validate the published MAE/SMAPE table, long-run convergence,
  inference-time claims or clinical deployment suitability.
- A full 100-trial GPU experiment is **not required** to assess the original
  repository audit or provide this code as a paper-aligned, openly qualified
  reference implementation. Do not claim numerical reproduction without it.

## Reviewer-facing description

> This repository preserves the original research code and anonymized phase
> sequences and provides a corrected TensorFlow evaluation runner reusing the
> archived model architectures. The corrected runner separates validation
> from outer test procedures, supports the paper's 100-trial/three-fold
> schedule, uses masked SMAPE, implements the reported two-model ensemble, and
> preserves unshuffled training batches. Unit tests and small real-data
> integration runs were performed. The original published results have not
> been independently reproduced, and architecture-level prefix causality is
> not established for the archived convolutional models.
