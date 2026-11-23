"""Fold preprocessing, masked SMAPE training and evaluation for ETC models.

The model definitions live in utils.models. This module handles procedures as
whole units and reserves the outer test fold for evaluation only.
"""
from __future__ import annotations

import numpy as np

PADDING = -1.0
TARGET_SCALE_SECONDS = 600.0


def split_train_validation(train_indices, *, fraction=0.2, seed=42):
    """Select a validation subset inside the outer training fold."""
    original = np.asarray(train_indices, dtype=int)
    if original.ndim != 1 or len(np.unique(original)) != len(original) or len(original) < 2:
        raise ValueError('At least two distinct training procedure indices are required')
    if not 0 < fraction < 1:
        raise ValueError('Validation fraction must be between zero and one')
    order = np.random.default_rng(seed).permutation(original)
    n_val = max(1, min(len(order) - 1, int(np.ceil(fraction * len(order)))))
    return np.sort(order[n_val:]), np.sort(order[:n_val])


def prepare_fold_arrays(X, Y, train, validation, test):
    """Fit input normalization on training procedures, retaining padding."""
    x = np.asarray(X)
    y = np.asarray(Y)
    if x.ndim != 3 or x.shape != y.shape or x.shape[-1] != 1:
        raise ValueError('X and Y must be aligned (procedures, timestamps, 1) arrays')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite X or Y values')
    valid_x = x[:, :, 0] != PADDING
    valid_y = y[:, :, 0] != PADDING
    if not np.array_equal(valid_x, valid_y) or not valid_y.any(axis=1).all():
        raise ValueError('X/Y valid timestamp masks do not match')
    if np.any(valid_y[:, 1:] & ~valid_y[:, :-1]):
        raise ValueError('Padding must be a suffix')
    if np.any(y[:, :, 0][valid_y] < 0):
        raise ValueError('Remaining time must be nonnegative')
    tr, va, te = [np.asarray(indices, dtype=int) for indices in (train, validation, test)]
    if len(set(tr) & set(va)) or len(set(tr) & set(te)) or len(set(va) & set(te)):
        raise ValueError('Training, validation and test procedures overlap')
    train_values = x[tr][x[tr] != PADDING]
    low, high = float(train_values.min()), float(train_values.max())
    width = high - low if high > low else 1.0

    def xx(indices):
        values = x[indices]
        # Genuine phase values are strictly positive, distinct from padding.
        return np.where(values == PADDING, PADDING, 1.0 + (values - low) / width).astype(np.float32)

    def yy(indices):
        values = y[indices]
        return np.where(values == PADDING, PADDING, values / TARGET_SCALE_SECONDS).astype(np.float32)

    return xx(tr), xx(va), xx(te), yy(tr), yy(va), yy(te)


class TargetScale:
    """Inverse scaling with -1 retained as the padding sentinel."""

    def inverse_transform(self, array):
        values = np.asarray(array)
        return np.where(values == PADDING, PADDING, values * TARGET_SCALE_SECONDS)


def smape(y_true, y_pred):
    """Paper-style SMAPE (0--100%) at valid timestamps; 0/0 contributes 0."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.shape != p.shape:
        raise ValueError('SMAPE arrays have different shapes')
    keep = y != PADDING
    if not np.any(keep):
        raise ValueError('No valid timestamps')
    t, q = y[keep], p[keep]
    if not np.isfinite(t).all() or not np.isfinite(q).all():
        raise ValueError('Nonfinite SMAPE values')
    den = np.abs(t) + np.abs(q)
    terms = np.divide(np.abs(t-q), den, out=np.zeros_like(den), where=den > 0)
    return float(100.0 * np.mean(terms))


def masked_smape_tf(y_true, y_pred):
    """Per-procedure TensorFlow SMAPE, omitting all -1 target timestamps."""
    import tensorflow as tf
    y_true = tf.cast(y_true, y_pred.dtype)
    keep = tf.not_equal(y_true, tf.cast(PADDING, y_pred.dtype))
    t = tf.where(keep, y_true, tf.zeros_like(y_true))
    p = tf.where(keep, y_pred, tf.zeros_like(y_pred))
    ratio = tf.math.divide_no_nan(tf.abs(p - t), tf.abs(t) + tf.abs(p))
    n = tf.reduce_sum(tf.cast(keep, y_pred.dtype), axis=(1, 2))
    return 100.0 * tf.math.divide_no_nan(tf.reduce_sum(ratio, axis=(1, 2)), n)


def train_model(model, X_train, y_train, X_val, y_val, *, epochs, batch_size):
    """Train without batch shuffling and select weights only on validation.

    Mirrors the paper's maximum 1000 epochs, early-stopping patience 100,
    and learning-rate reduction patience 50 with a 0.001 improvement margin.
    Training uses a masked SMAPE GradientTape objective rather than the MAE
    default of the reusable architecture factories.
    """
    import tensorflow as tf

    if epochs < 1 or batch_size < 1 or len(X_train) == 0 or len(X_val) == 0:
        raise ValueError('Positive epochs/batch size and nonempty sets required')
    optimizer = model.optimizer
    best_val = float('inf')
    best_weights = None
    since_best = 0
    since_lr = 0
    completed = 0
    for epoch in range(epochs):
        for start in range(0, len(X_train), batch_size):
            xb = tf.convert_to_tensor(X_train[start:start+batch_size], dtype=tf.float32)
            yb = tf.convert_to_tensor(y_train[start:start+batch_size], dtype=tf.float32)
            with tf.GradientTape() as tape:
                pred = model(xb, training=True)
                loss = tf.reduce_mean(masked_smape_tf(yb, pred))
            if not bool(tf.math.is_finite(loss)):
                raise FloatingPointError('Nonfinite training loss')
            grads = tape.gradient(loss, model.trainable_variables)
            updates = [(g, w) for g, w in zip(grads, model.trainable_variables) if g is not None]
            if not updates:
                raise RuntimeError('Model has no trainable gradients')
            optimizer.apply_gradients(updates)

        # Validation is also batched to limit memory use for the CNN models.
        scores = []
        weights = []
        for start in range(0, len(X_val), batch_size):
            xb = tf.convert_to_tensor(X_val[start:start+batch_size], dtype=tf.float32)
            yb = tf.convert_to_tensor(y_val[start:start+batch_size], dtype=tf.float32)
            scores.append(float(tf.reduce_sum(masked_smape_tf(yb, model(xb, training=False))).numpy()))
            weights.append(len(xb))
        val_smape = sum(scores) / sum(weights)
        if not np.isfinite(val_smape):
            raise FloatingPointError('Nonfinite validation SMAPE')
        completed = epoch + 1
        if val_smape < best_val - 0.001:
            best_val = val_smape
            best_weights = [a.copy() for a in model.get_weights()]
            since_best = 0
            since_lr = 0
        else:
            since_best += 1
            since_lr += 1
            if since_lr >= 50:
                lr = optimizer.learning_rate
                lr.assign(tf.maximum(lr * 0.5, tf.cast(1e-7, lr.dtype)))
                since_lr = 0
            if since_best >= 100:
                break
    if best_weights is None:
        raise RuntimeError('No valid validation checkpoint')
    model.set_weights(best_weights)
    return {'epochs_run': completed, 'best_validation_smape_percent': best_val}
