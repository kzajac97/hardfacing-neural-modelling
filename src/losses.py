import numpy as np
import tensorflow as tf


class AdaptiveLoss(tf.keras.losses.Loss):
    """Simplified adaptive loss function, described in https://arxiv.org/abs/1701.03077"""

    def __init__(self, alpha: float, scale: float) -> None:
        """
        :param alpha: shape parameter
        :param scale: scale parameter
        """
        super(AdaptiveLoss, self).__init__()

        self.alpha = alpha
        self.scale = scale

    def _special_case(self, alpha):
        return {
            2.0: self._special_case_L2,
            0.0: self._special_case_l1,
            -1.0 * np.inf: self._special_case_l_infty,
        }.get(alpha, self._general_case)

    def _special_case_l2(self, y_true, y_pred):
        return 0.5 * tf.math.pow((y_true - y_pred / self.scale), 2)

    def _special_case_l1(self, y_true, y_pred):
        return tf.math.log(0.5 * tf.math.pow((y_true - y_pred / self.scale), 2) + 1)

    def _special_case_l_infty(self, y_true, y_pred):
        return 1 - tf.exp(-0.5 * tf.math.pow((y_true - y_pred / self.scale), 2))

    def _general_case(self, y_true, y_pred):
        return (
            tf.abs(self.alpha - 2)
            / self.alpha
            * (
                tf.math.pow(
                    1 + tf.math.pow((y_true - y_pred / self.scale), 2) / tf.abs(self.alpha - 2), 0.5 * self.alpha
                )
                - 1
            )
        )

    def __call__(self, y_true, y_pred, *args, **kwargs):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        return self._special_case(self.alpha)(y_true, y_pred)
