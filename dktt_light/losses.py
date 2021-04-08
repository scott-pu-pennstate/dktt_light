import tensorflow as tf


class PaddedBinaryCrossentropyLoss(tf.keras.losses.Loss):
    def __init__(self, smoothing=0.):
        super(PaddedBinaryCrossentropyLoss, self).__init__()
        self.smoothing = smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype='float32')
        weights = tf.cast(tf.not_equal(y_true, 0.), tf.float32)
        y_true = tf.maximum(y_true - 1., 0.)
        y_true = y_true * (1.0 - self.smoothing) + 0.5 * self.smoothing
        bxentropy = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=False)

        bxentropy *= weights
        loss = tf.reduce_sum(bxentropy) / (tf.reduce_sum(weights) + 1e-4)
        return loss
