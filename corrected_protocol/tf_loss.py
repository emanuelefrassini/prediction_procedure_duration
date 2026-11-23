"""TensorFlow 2.10 masked SMAPE for the corrected clinical rerun protocol."""


def masked_smape_tf(y_true, y_pred):
    """Per-procedure masked 0..100 SMAPE, compatible with Keras reductions.

    Returns a vector of shape (batch,), averaging only valid time points in
    each procedure, so long cases do not dominate the objective. A padding
    suffix is represented by -1 in y_true. Padded y_pred has no effect.
    """
    import tensorflow as tf

    y_true = tf.cast(y_true, y_pred.dtype)
    keep = tf.not_equal(y_true, tf.cast(-1.0, y_pred.dtype))
    t = tf.where(keep, y_true, tf.zeros_like(y_true))
    p = tf.where(keep, y_pred, tf.zeros_like(y_pred))
    ratio = tf.math.divide_no_nan(tf.abs(p - t), tf.abs(p) + tf.abs(t))
    counts = tf.reduce_sum(tf.cast(keep, y_pred.dtype), axis=(1, 2))
    per_case = tf.math.divide_no_nan(tf.reduce_sum(ratio, axis=(1, 2)), counts)
    return 100.0 * per_case
