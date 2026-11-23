# Model experiments

[`PAPER_ETC.py`](PAPER_ETC.py) is the main multi-architecture experiment for the published benchmark. It reads the numerical data in `../data/`, calls the original model factories from `../utils/models.py`, and writes a timestamped run under `models/runs/`.

## Experiment scripts

| Script | Purpose |
|---|---|
| [`PAPER_ETC.py`](PAPER_ETC.py) | Five-architecture evaluation, fixed two-model ensemble, multiple sampling intervals and procedure-level cross-validation |
| `lstm_ETC.py` | LSTM-specific historical experiments |
| `LSTM_Attention_ETC.py` | Attention-LSTM experiments |
| `lstm_fcn_ETC.py` | LSTM-FCN experiments |
| `transformer_ETC.py` | Transformer experiments |
| `inceptiontime_ETC.py` | InceptionTime experiments |
| `*_hpsearch.py` | Architecture-specific hyperparameter-search code |
| `models.py` | Additional historical model definitions; the main runner imports from `../utils/models.py` |
| `load_data.py` | Historical data-loader helper for an NPZ file |

Individual architecture scripts may depend on workstation-specific data paths and historical output folders. **Use `PAPER_ETC.py` as the primary entry point** for the current repository's `X.csv`/`Y.csv` experiment. Do not assume the auxiliary scripts use the same preprocessing and validation logic.

The model factory implementations and their capacity parameters are documented in [`../docs/methods.md`](../docs/methods.md). Runtime setup: [`../docs/reproducibility.md`](../docs/reproducibility.md).
