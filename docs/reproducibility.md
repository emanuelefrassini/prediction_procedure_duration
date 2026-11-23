# Experiment setup and reproducibility

The main experiment is `models/PAPER_ETC.py`. It reads anonymized phase/target arrays from `data/X.csv` and `data/Y.csv` (or `ETC_DATA_DIR`) and writes trial/fold/model results to `models/runs/`.

## Environment

The experiment was developed for TensorFlow/Keras 2.10 and Python 3.10. The dependency specification in `requirements-legacy-py310.txt` supplies a compatible starting point. Python 3.14 is not supported by TensorFlow 2.10.

```bash
python3.10 -m venv .venv-tf210
source .venv-tf210/bin/activate
python -m pip install -r requirements-legacy-py310.txt
python -m unittest discover -s tests -p 'test_*.py' -v
```

The full training run is `python -m models.PAPER_ETC`. Small diagnostic runs can set `ETC_TRIALS`, `ETC_EPOCHS` and `ETC_INTERVALS` environment variables. It is configured for 100 trials, three procedure-level outer folds, five sampling intervals and five base architectures, with up to 1,000 epochs. An ensemble is calculated from the InceptionTime and LSTM-FCN test predictions.

## Preprocessing and evaluation

- A procedure stays entirely in one outer partition per fold.
- An inner validation subset is selected from the outer training partition (20% of its procedures), never from the test partition.
- Input scaling is fitted to valid timestamps of the inner training subset. Padding (`-1`) is retained throughout preprocessing.
- Remaining-time targets are converted from seconds with a fixed scale of 600 seconds; padding is unchanged.
- The training objective is per-procedure SMAPE with factor 100. It omits padded timestamps and assigns zero when both values are zero.
- Early stopping uses validation SMAPE, patience 100, and best-weight restoration; the learning rate is halved after 50 non-improving epochs (improvement threshold 0.001).
- Adam is used with the architecture-specific learning rates. Batches are kept in fixed order rather than shuffled between epochs.
- MAE in minutes and SMAPE in percent are calculated from predictions converted back to seconds. Procedure-level averages assign equal weight to each case.
- The ensemble is the unweighted average of InceptionTime and LSTM-FCN predictions. No test labels are used to set ensemble weights.

## Scope

The model factories and hyperparameters remain those from the repository's TensorFlow research scripts. The validation partition and padded-target objective make the evaluation procedure explicit and avoid model selection on held-out test data. These choices do not imply bitwise identity to the training conditions that produced the published table. The quoted paper results are not presented as newly recomputed values.

Some of the original convolutional blocks use centered padding and the LSTM-FCN model uses sequence-wise pooling. These choices follow the existing architecture code; they are not guaranteed to be strictly causal at each time point. An online implementation would require a separate architectural and evaluation study.
