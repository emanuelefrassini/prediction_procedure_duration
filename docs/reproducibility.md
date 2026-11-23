# Environment, execution and reproducibility

The primary experiment is [`models/PAPER_ETC.py`](../models/PAPER_ETC.py). It reads the anonymized phase/target data from `data/X.csv` and `data/Y.csv`, or from an equivalent directory specified with `ETC_DATA_DIR`, and writes its outputs to `models/runs/`.

## Software environment

The archived model factories use **TensorFlow/Keras 2.10** and a compatible **Python 3.10** interpreter. The pinned values in `requirements-legacy-py310.txt` provide a compatible environment candidate. They are not a copy of the original research machine's historical lockfile; exact historical dependency versions and GPU runtime cannot be inferred from the repository.

```bash
python3.10 -m venv .venv-tf210
source .venv-tf210/bin/activate
python -m pip install -r requirements-legacy-py310.txt
python scripts/check_experiment.py --training
python -m unittest discover -s tests -p 'test_*.py' -v
```

Python 3.14 is not supported by TensorFlow 2.10. GPU acceleration is optional for checking the code; a complete 100-trial experiment can require substantial compute and memory. TensorFlow 2.10 GPU support on Linux requires compatible system CUDA/cuDNN libraries that are **not** installed by this requirements file.

## Check the dataset

The two CSV files are public anonymized numerical procedure sequences; they are *not* raw clinical videos. Expected shape is **222 rows × 7,573 columns**, with `-1` as trailing padding. You can check dimensions and padding without displaying procedure data:

```bash
python scripts/inspect_dataset.py --data-dir data
```

The main runner loads these files by default. To use copies in another location:

```bash
ETC_DATA_DIR=/path/to/approved/data python -m models.PAPER_ETC
```

The outer folds, model specifications and sampling intervals are described in the [methods guide](methods.md).

## Run the experiment

From the repository root, with the TensorFlow environment active:

```bash
python -m models.PAPER_ETC
```

The default is **100 trials**, **3 procedure-level outer folds**, **5 temporal sampling intervals** and **5 fitted architectures**. An InceptionTime/LSTM-FCN ensemble is calculated from their predictions. Maximum training duration is 1,000 epochs per model with validation-based early stopping.

To check that execution works without running the full benchmark, override the trial count, maximum epochs and sampling intervals using environment variables:

```bash
ETC_TRIALS=1 ETC_EPOCHS=1 ETC_INTERVALS=120 python -m models.PAPER_ETC
```

**This is a software smoke test, not a predictive-performance estimate.** The full trial configuration is not required to verify parsing, model construction, the loss function or output writing. Training outputs are stored in a timestamped run directory beneath `models/runs/`, which is ignored by Git.

## Evaluation conventions

- **Partition unit:** complete procedures, not individual timestamps.
- **Cross-validation:** 100 repeated trials of three outer folds. A disjoint inner validation subset of 20% is selected from each outer training partition.
- **Preprocessing:** training-only scaling of valid phase values; target scaling by a fixed 600-second constant; `-1` padding in targets is excluded from the loss and scores.
- **Training objective:** per-procedure symmetric mean absolute percentage error (SMAPE), with multiplier 100 and zero contribution for `0/0`.
- **Optimization:** Adam with model-specific learning rates, unshuffled batches, up to 1,000 epochs, 100-epoch stopping patience and 50-epoch learning-rate adjustment patience.
- **Final evaluation:** each test fold is evaluated after validation-only model selection. MAE is reported in minutes, SMAPE in percent; the ensemble averages InceptionTime and LSTM-FCN predictions equally.

The maintained code's explicit validation partition and padding treatment may differ from training details of some original archived experiment scripts. Neither the automated tests nor a short local run establish reproduction of the scores published in 2025. For any scientific comparison, disclose the precise code revision and run settings.

## Inspect run outputs

Use [`scripts/summarize_results.py`](../scripts/summarize_results.py) on a completed or partial run folder. It reads only aggregate fold-level score files, not the CSV dataset or individual predictions:

```bash
python scripts/summarize_results.py models/runs/runs_PAPER_YYYYMMDD-HHMMSS
```

The output counts completed folds so that a partial experiment cannot silently be mistaken for a complete 100-trial evaluation. See [output structure and metrics](results.md) and the [historical analysis scripts](../tools/README.md).

## Interpretation

The paper's reported Table 2 results in the main README are quoted from the publication; they are **not** freshly computed performance claims. Model outputs are intended for retrospective research. Some historical convolution/pooling components can access later timestamps, and a strictly causal online forecasting implementation would require separate methodological development and validation. The public dataset and this code are not a clinical decision-support product.
