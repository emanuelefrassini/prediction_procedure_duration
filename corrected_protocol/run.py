"""Auditable rerun protocol using archived Keras model factories.

This corrects evaluation splitting, padding-aware training, SMAPE and ensemble
construction. It does NOT assert that these settings or the archived model
implementations were used to generate the published 2025 results. The model
architectures are historical and have not been audited for prefix causality.

Dry-run (no TensorFlow or data needed):
    python -m corrected_protocol

A very small actual experiment (Python 3.10 + TensorFlow 2.10):
    python -m corrected_protocol --execute --trials 1 --epochs 2 \\
      --intervals 60 --models inceptiontime lstm_fcn
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform

import numpy as np

from .protocol import (
    MODEL_NAMES, SAMPLING_INTERVALS, TARGET_SCALE_SECONDS,
    fixed_two_model_ensemble, fold_plans, prepare_arrays,
)

ROOT = Path(__file__).resolve().parents[1]


def read_authorized_data(directory: Path):
    """Read only during --execute; never serialize procedure sequences."""
    x = np.loadtxt(directory / 'X.csv', delimiter=',', dtype=np.float32)
    y = np.loadtxt(directory / 'Y.csv', delimiter=',', dtype=np.float32)
    if x.shape != y.shape or x.ndim != 2 or x.shape != (222, 7573):
        raise ValueError('Expected aligned numeric CSV arrays of shape (222, 7573)')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Dataset contains non-finite values')
    return x[..., None], y[..., None]


def build_architecture(name: str, length: int, epochs: int):
    """Use archived factories, but discard their MAE loss and callbacks."""
    from utils.models import (
        model_LSTM, model_LSTM_Attention, model_LSTM_FCN,
        model_transformer, model_INCEPTIONTIME,
    )
    if name == 'lstm':
        return model_LSTM(UNITS=128, DROPOUT_RATE=0.1, LEARNING_RATE=0.001, EPOCHS=epochs)[0]
    if name == 'lstm_attention':
        return model_LSTM_Attention(UNITS=128, DROPOUT_RATE=0.1, LEARNING_RATE=0.001, EPOCHS=epochs)[0]
    if name == 'lstm_fcn':
        return model_LSTM_FCN(length, 1, UNITS=256, DROPOUT_RATE=0.1, LEARNING_RATE=0.001, EPOCHS=epochs)[0]
    if name == 'transformer':
        return model_transformer(length, 1, LEARNING_RATE=0.001, EPOCHS=epochs,
                                 head_size=16, ff_dim=4, num_transformer_blocks=2)[0]
    if name == 'inceptiontime':
        return model_INCEPTIONTIME(length, 1, LEARNING_RATE=0.0001,
                                   EPOCHS=epochs, BATCH_SIZE=2, stride=1,
                                   activation='linear', nb_filters=30,
                                   use_residual=True, depth=8, kernel_size=41,
                                   use_bottleneck=False, bottleneck_size=32)[0]
    raise ValueError('Unknown model: ' + name)


BATCH_SIZES = {'lstm': 8, 'lstm_attention': 8, 'lstm_fcn': 4,
               'transformer': 2, 'inceptiontime': 2}


def unshuffled_procedure_batches(n_procedures: int, batch_size: int):
    """Yield consecutive case indices without per-epoch shuffling.

    The 2025 paper specifies non-shuffled batches. The upstream fold plan
    determines which procedures are in training; their order is left intact.
    This changes neither outer fold assignment nor validation/test isolation.
    """
    if n_procedures < 1 or batch_size < 1:
        raise ValueError('n_procedures and batch_size must be positive')
    for start in range(0, n_procedures, batch_size):
        yield np.arange(start, min(start + batch_size, n_procedures))


def fit_one(name, x_train, y_train, x_val, y_val, x_test, *, epochs, seed):
    """Explicit masked-SMAPE training with independent validation-only selection.

    A GradientTape loop avoids ambiguity from automatic Keras sequence-mask
    sample weighting on the archived architectures. No test target is supplied.
    Early stopping and learning-rate scheduling have fresh state per model.
    """
    import tensorflow as tf
    from .tf_loss import masked_smape_tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_architecture(name, x_train.shape[1], epochs)
    optimizer = model.optimizer  # constructed by the archived factory
    early_patience = max(2, epochs // 10)
    lr_patience = max(2, epochs // 20)
    best_val = float('inf')
    best_weights = None
    no_improvement = 0
    lr_no_improvement = 0
    epoch_count = 0

    for epoch in range(epochs):
        # The paper states that batches were constructed without shuffling.
        # Each item is one whole procedure; no outer test case is used here.
        for idx in unshuffled_procedure_batches(len(x_train), BATCH_SIZES[name]):
            batch_x = tf.convert_to_tensor(x_train[idx], dtype=tf.float32)
            batch_y = tf.convert_to_tensor(y_train[idx], dtype=tf.float32)
            with tf.GradientTape() as tape:
                prediction = model(batch_x, training=True)
                objective = tf.reduce_mean(masked_smape_tf(batch_y, prediction))
            gradients = tape.gradient(objective, model.trainable_variables)
            pairs = [(g, v) for g, v in zip(gradients, model.trainable_variables)
                     if g is not None]
            if not pairs:
                raise RuntimeError(f'No trainable gradients for {name}')
            optimizer.apply_gradients(pairs)

        # Validation-only model selection: outer test arrays are not used.
        val_prediction = model(tf.convert_to_tensor(x_val, dtype=tf.float32),
                               training=False)
        val_loss = float(tf.reduce_mean(masked_smape_tf(
            tf.convert_to_tensor(y_val, dtype=tf.float32), val_prediction)).numpy())
        epoch_count = epoch + 1
        if not np.isfinite(val_loss):
            raise FloatingPointError(f'Non-finite validation SMAPE for {name}')
        if val_loss < best_val - 0.001:
            best_val = val_loss
            best_weights = [v.copy() for v in model.get_weights()]
            no_improvement = 0
            lr_no_improvement = 0
        else:
            no_improvement += 1
            lr_no_improvement += 1
            if lr_no_improvement >= lr_patience:
                lr_variable = optimizer.learning_rate
                lr_variable.assign(tf.maximum(lr_variable * 0.5,
                                              tf.cast(1e-7, lr_variable.dtype)))
                lr_no_improvement = 0
            if no_improvement >= early_patience:
                break

    if best_weights is None:
        raise RuntimeError(f'No valid epoch found for {name}')
    model.set_weights(best_weights)
    prediction = model(tf.convert_to_tensor(x_test, dtype=tf.float32),
                       training=False).numpy().astype(np.float64)
    metadata = {'epochs_run': epoch_count,
                'best_validation_smape_percent': best_val}
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return prediction, metadata

def score(y_true_seconds, y_pred_seconds):
    from phase_etc.metrics import per_procedure_metrics
    summary = per_procedure_metrics(y_true_seconds, y_pred_seconds)
    return {
        'macro_mae_minutes': summary['macro_mae_minutes'],
        'macro_smape_percent_0_to_100': summary['macro_smape_percent'],
    }


def evaluate_fold(data_x, data_y, trial_fold, interval, models, *, epochs, seed,
                  trainer=None):
    """Train on train/validation only; reserve outer test for final scoring.

    ``trainer`` injection permits a data-free integration test that proves
    the test targets are never handed to any model fit.
    """
    if trainer is None:
        trainer = fit_one
    sampled_x = data_x[:, ::interval, :]
    sampled_y = data_y[:, ::interval, :]
    tr, va, te = trial_fold.train, trial_fold.validation, trial_fold.test
    (xx, yy) = prepare_arrays(sampled_x[tr], sampled_x[va], sampled_x[te],
                              sampled_y[tr], sampled_y[va], sampled_y[te])
    xtr, xva, xte = xx
    ytr, yva, _ = yy
    predictions_seconds, report = {}, {}
    for model_number, name in enumerate(models):
        pred_scaled, meta = trainer(name, xtr, ytr, xva, yva, xte,
                                    epochs=epochs,
                                    seed=seed + 100000 * trial_fold.trial
                                         + 1000 * trial_fold.fold
                                         + 10 * interval + model_number)
        pred_seconds = pred_scaled * TARGET_SCALE_SECONDS
        predictions_seconds[name] = pred_seconds
        report[name] = {**score(sampled_y[te], pred_seconds), **meta}
    if 'inceptiontime' in predictions_seconds and 'lstm_fcn' in predictions_seconds:
        report['ensemble'] = score(sampled_y[te], fixed_two_model_ensemble(predictions_seconds))
    return {
        'trial': trial_fold.trial, 'fold': trial_fold.fold,
        'interval_seconds': interval,
        'n_train': len(tr), 'n_validation': len(va), 'n_test': len(te),
        'models': report,
    }


def run(args):
    import tensorflow as tf  # only required with --execute

    data_x, data_y = read_authorized_data(args.data_dir)
    n = data_x.shape[0]
    record = {
        'scope': 'corrected evaluation rerun; not a verified reproduction of published numbers',
        'python': platform.python_version(), 'tensorflow': tf.__version__,
        'configuration': {
            'trials': args.trials, 'folds': args.folds, 'epochs': args.epochs,
            'seed': args.seed, 'validation_fraction': args.validation_fraction,
            'intervals_seconds': args.intervals, 'models': args.models,
            'target_scale_seconds': TARGET_SCALE_SECONDS,
            'loss': 'masked per-procedure SMAPE with factor 100',
            'ensemble': 'fixed mean of inceptiontime and lstm_fcn',
        },
        'results': [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for trial_fold in fold_plans(n, trials=args.trials, folds=args.folds,
                                 seed=args.seed,
                                 validation_fraction=args.validation_fraction):
        for interval in args.intervals:
            fold_record = evaluate_fold(data_x, data_y, trial_fold, interval,
                                        args.models, epochs=args.epochs, seed=args.seed)
            record['results'].append(fold_record)
            # Preserve partial aggregate results after each completed fold.
            tmp = args.output.with_suffix('.tmp')
            tmp.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            tmp.replace(args.output)
            print(f'trial={trial_fold.trial:03d} fold={trial_fold.fold} '
                  f'interval={interval} s: {list(fold_record["models"])}', flush=True)
    return record


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true',
                   help='Explicitly run TF training using authorized local CSVs')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    p.add_argument('--output', type=Path, default=ROOT / 'models' / 'runs' / 'corrected_summary.json')
    p.add_argument('--trials', type=int, default=100)
    p.add_argument('--folds', type=int, default=3)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--validation-fraction', type=float, default=0.2)
    p.add_argument('--intervals', nargs='+', type=int, default=list(SAMPLING_INTERVALS))
    p.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=list(MODEL_NAMES))
    args = p.parse_args(argv)
    if args.trials < 1 or args.folds < 2 or args.epochs < 1 or args.folds > 222:
        p.error('trials/epochs >= 1 and 2 <= folds <= 222 required')
    if not 0 < args.validation_fraction < 1:
        p.error('validation fraction must be in (0,1)')
    if not args.intervals or any(x not in SAMPLING_INTERVALS for x in args.intervals):
        p.error('intervals must be among 5, 10, 30, 60, 120')
    if len(set(args.intervals)) != len(args.intervals) or len(set(args.models)) != len(args.models):
        p.error('duplicate intervals/models are not allowed')
    return args


def main(argv=None):
    args = parse_args(argv)
    fits = args.trials * args.folds * len(args.intervals) * len(args.models)
    print(f'Planned model fits: {fits} '
          f'({args.trials} trials × {args.folds} folds × '
          f'{len(args.intervals)} intervals × {len(args.models)} architectures)')
    if not args.execute:
        print('DRY RUN: No patient-derived data accessed and no model trained. '
              'Pass --execute to train using the local data directory.')
        return
    run(args)
    print('Corrected rerun output:', args.output)


if __name__ == '__main__':
    main()
