# Causal remaining-time benchmark (synthetic data)

This directory contains a new, independent CPU-executable PyTorch benchmark inspired by the procedure-duration study linked in the root README. It does not replace the original TensorFlow/Keras experiments, reconstruct their exact hyperparameters, or reproduce their clinical metrics. No procedure-derived data, patient information, or pretrained model is needed.

## What is implemented

Five compact sequence-to-sequence model families take a categorical phase sequence and elapsed sample index: LSTM, LSTM with causal self-attention, LSTM-FCN with causal convolutions, an InceptionTime-inspired multiscale causal convolution block, and a causal Transformer. Each predicts nonnegative remaining seconds at every valid timestamp. The architecture names denote the families, not identical implementations of the original models. For example, the attention mechanism and convolutional blocks differ from the historical networks, and the InceptionTime variant is deliberately shallow.

The synthetic generator produces monotonically ordered phases 0–13 with stochastic duration per phase and a shared case-level speed factor. Each target is the number of seconds until the last sampled timestamp. These artificial workflows are not a simulator of coronary angiography, do not contain phase repetitions or complex clinical event distributions, and do not support clinical conclusions.

## Evaluation protocol

- Outer shuffled folds split whole synthetic procedures, never individual timestamps. Each test fold is used only for final scoring.
- An independent validation subset is drawn from the training side of each fold. The lowest validation MAE determines the selected epoch; the test set is not consulted for this choice.
- Phase IDs use an embedding and the elapsed index is a known causal feature; the model is not given the total duration, final sequence length, or future phases. Recurrent layers run only forward; all self-attention and convolutions are causal. Right-padding is excluded from the objective. A test changes future inputs and truncates the sequence to check that predictions for the shared prefix agree.
- The objective is masked absolute error with a fixed 600-second numerical scale (not estimated from any data). The final outputs are converted back to seconds.
- A train-only mean remaining-time baseline is calculated for each phase. The two-network ensemble is fixed in advance as the arithmetic mean of the InceptionTime-inspired and LSTM-FCN outputs. No model selection is performed on test data.
- Report pooled MAE (all valid timestamps), macro MAE (each case has equal weight), and the paper-style SMAPE convention (100 times the mean absolute difference divided by the sum of absolute values, with zero/zero defined as zero). The report averages fold-level metrics equally; it is not a pooled estimate across folds or a confidence interval.

## Run from the repository root

Use a dedicated Python 3.11 or 3.12 environment. CI uses Python 3.11. The historical research environment is not needed.

```bash
python -m pip install 'torch>=2.2,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r modern_etc/requirements.txt
python -m pytest modern_etc/tests -q
python -m modern_etc --cases 90 --folds 3 --epochs 20 --seed 42 --width 32 --interval 30 --output /tmp/etc_synthetic.json
```

A shorter smoke run is `python -m modern_etc --cases 18 --epochs 1 --width 8 --output /tmp/etc_smoke.json`. Run `python -m modern_etc --help` for other options. The default is five epochs for a short demonstration. The runner uses CPU, one thread, seeded initialization and deterministic PyTorch algorithms. Exact floating-point results can vary across framework versions and platforms. It outputs a JSON manifest with split IDs, selected epochs, validation metrics, and synthetic test metrics. It does not save trained model weights or perform a hyperparameter search.

## Example results

The checked-in [JSON report](results/synthetic_example.json) was produced with 90 synthetic cases, three folds, 20 epochs, a 30-second sampling interval, seed 42 and width 32, using PyTorch 2.10.0+cpu. The numbers below are mean fold-level macro MAE, in seconds.

| Model | Macro MAE (s) |
| --- | ---: |
| Training-phase mean baseline | 225.72 |
| LSTM | 224.53 |
| LSTM with attention | 236.01 |
| LSTM-FCN | 218.21 |
| InceptionTime-inspired | 213.55 |
| Transformer | 231.85 |
| Fixed two-model ensemble | 214.23 |

Some models outperform the simple baseline on this run, while others do not. These results are a single artificial-data experiment, not a statistical comparison and not the paper's results. The benchmark also passes tests for split isolation, masked loss/metric conventions, causality, finite gradients and repeatability of a small training run. GitHub Actions is configured to test the new module without checking out the clinical CSVs.

## Next steps

Recover the exact authorized study configuration, adapt input handling for real sequences with appropriate phase annotations, test realistic missing/incorrect phases, repeat with multiple independent seeds, and add uncertainty estimates. Do not use the public repository's historical CSVs until redistribution and local usage permissions are confirmed. In particular, deleting a tracked file in a later commit does not erase its prior Git history.
