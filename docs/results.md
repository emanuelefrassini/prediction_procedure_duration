# Experiment outputs and result processing

`models/PAPER_ETC.py` writes to a timestamped folder under `models/runs/`. The convention is retained so that the original research tools can still locate fold-level files.

```text
models/runs/runs_PAPER_YYYYMMDD-HHMMSS/
└── trial_1/
    └── reduce_dim_60/
        └── fold_0/
            ├── Indices.npy
            ├── INCEPTIONTIME/
            │   ├── model/
            │   ├── model_summary.txt
            │   ├── mae_per_procedure.txt
            │   ├── SMAPE_per_procedure.txt
            │   ├── time.txt
            │   └── test_predictions/
            ├── LSTM/
            ├── LSTM_FCN/
            ├── LSTM_Attention/
            ├── transformer/
            └── Ensemble/
```

A complete 100-trial experiment has 100 trial folders and 3 test folds for each of the 5 sampling intervals. The saved model and prediction files can be large; `models/runs/` is ignored by Git.

## Fold-level metrics

The first line of `mae_per_procedure.txt` contains the mean of the procedure-specific MAEs in **minutes**. The first line of `SMAPE_per_procedure.txt` contains the mean of procedure-specific SMAPEs, using the paper's **0–100%** convention. Subsequent lines contain individual procedure metrics. Those files are preserved by the main script.

The `Indices.npy` file stores the training, validation and test procedure indices for each fold. The ensemble uses only the two constituent model predictions (InceptionTime and LSTM-FCN), after those models have been trained separately.

## Summarize results without retraining

Run from the repository root, using an existing run folder:

```bash
python scripts/summarize_results.py models/runs/runs_PAPER_YYYYMMDD-HHMMSS
python scripts/summarize_results.py models/runs/runs_PAPER_YYYYMMDD-HHMMSS --format csv
python scripts/summarize_results.py models/runs/runs_PAPER_YYYYMMDD-HHMMSS --format json --output /tmp/etc_summary.json
```

The summary tool uses only the *aggregate first line* of each score file. It does not load `X.csv`, `Y.csv`, per-procedure predictions, TensorFlow or trained models. It reports, for each model and sampling interval, the number of **complete model-fold records** and their arithmetic mean and sample standard deviation.

If a model folder is only partially written, it is omitted from the statistics and reported as incomplete. **The summary is a descriptive average of available folds, not a reconstruction of the paper's across-trial uncertainty estimates.** In particular, a partially completed run must not be labeled as a 100-trial outcome.

## Original analysis scripts

The historical scripts in [`tools/`](../tools/README.md) contain result extraction, visualisation and ensemble-analysis workflows. Some use absolute Windows paths referencing the original study workstation or Excel summaries; adjust those paths to an actual local run before using them. They are separate from the standard-library summarizer above.
