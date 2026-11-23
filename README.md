# Clinical procedure duration prediction

Code and anonymized phase-sequence data accompanying the paper:

Frassini, E., Vijfvinkel, T. S., Butler, R. M., van der Elst, M., Hendriks, B. H. W., & van den Dobbelsteen, J. J. (2025). **Deep learning methods for clinical workflow phase-based prediction of procedure duration: a benchmark study.** *Computer Assisted Surgery, 30*(1), 2466426. https://doi.org/10.1080/24699322.2025.2466426

The study compares deep-learning models for estimating the remaining time until the end of coronary angiography procedures from manually annotated workflow phases. The inputs are phase sequences, not raw videos. The dataset contains **222 procedures** annotated with **14 clinical phases**.

## Methods

The main experiment is `models/PAPER_ETC.py`. It uses the five architecture definitions in `utils/models.py` and evaluates their predictions together with the average of **InceptionTime and LSTM-FCN**:

- LSTM
- LSTM with attention
- LSTM-FCN
- Transformer
- InceptionTime
- Ensemble: equal-weight average of InceptionTime and LSTM-FCN

The experiment is configured for **100 trials**, **three cross-validation folds per trial**, and sampling intervals of **5, 10, 30, 60 and 120 seconds**. The outer folds separate whole procedures. Each training fold is partitioned into training and validation procedures; validation drives early stopping and learning-rate adjustments, and the outer test fold is used only for final evaluation.

Training uses Adam and a padding-aware SMAPE objective with the paper's **0–100%** normalization. The maximum number of epochs is 1,000; early stopping has a patience of 100 epochs and the learning rate is halved after 50 epochs without sufficient validation improvement. Training batches are not shuffled. Model architecture definitions and hyperparameters in `utils/models.py` and `models/PAPER_ETC.py` remain the ones used in the repository's research scripts.

The full experiment is computationally expensive. See [reproducibility instructions](docs/reproducibility.md) for the environment and data handling. Smaller local test runs are useful for code checks but are not substitutes for a 100-trial benchmark.

## Data

`data/X.csv` contains sampled clinical phase inputs and `data/Y.csv` the corresponding remaining-time targets. Both files are publicly available in this repository and contain anonymized procedure-derived arrays. The matching arrays have shape `(222, 7573)`, padded with `-1` after the valid end of each procedure. The experiment reads them from `data/` by default; set `ETC_DATA_DIR` to use another directory containing `X.csv` and `Y.csv`.

## Repository structure

| Directory | Contents |
|---|---|
| `models/` | Main experiment and model-specific TensorFlow/Keras research scripts |
| `utils/` | Model definitions, data preparation, training and evaluation helpers |
| `data/` | Anonymized phase and remaining-time sequences |
| `tools/` | Result processing, ensemble analysis and paper plots |
| `tests/` | Data-handling and experiment-configuration checks |
| `docs/` | Environment and reproducibility guidance |

## Results

The following values are quoted from Table 2 of the publication at a sampling interval of **60 seconds**. They are **not** new measurements from the current code checkout.

| Architecture | MAE (min) | SMAPE (%) |
|---|---:|---:|
| InceptionTime | 0.53 ± 0.09 | 5.8 ± 1.2 |
| LSTM | 12.06 ± 2.44 | 44.5 ± 19.3 |
| LSTM-FCN | 2.35 ± 0.92 | 12.3 ± 5.3 |
| Transformer | 7.13 ± 2.31 | 19.3 ± 8.2 |
| LSTM-Attention | 17.33 ± 3.54 | 75.6 ± 21.1 |
| Ensemble | 1.32 ± 0.76 | 7.2 ± 1.9 |

The reported single-center results should not be interpreted as external validation or performance guarantees. The architecture definitions retain the original time-series convolution and attention choices; strictly online prefix-causal behavior is not claimed.

## Installation and checks

The TensorFlow experiment uses Python 3.10 and TensorFlow 2.10. See [`requirements.txt`](requirements.txt) and [`docs/reproducibility.md`](docs/reproducibility.md). Lightweight checks can be run without TensorFlow:

```bash
python -m pip install -r requirements-validation.txt
python -m unittest discover -s tests -p 'test_*.py' -v
```

The training script writes results under `models/runs/`. To start the full experiment in the TensorFlow environment:

```bash
python -m models.PAPER_ETC
```

Running the full set of trials requires substantial compute resources. To do a limited local integration check without changing the default experiment settings, set `ETC_TRIALS=1 ETC_EPOCHS=1 ETC_INTERVALS=120` before the command. This still fits the five base models over three folds and does not estimate fully trained model performance.
