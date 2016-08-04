import tensorflow as tf
import numpy as np
from .rbm import RBM
from .util import sample_prob


class BBRBM(RBM):
    def __init__(self, n_visible, n_hidden, *, learning_rate=0.1, momentum=0.5):
        RBM.__init__(self, n_visible, n_hidden, learning_rate=learning_rate, momentum=momentum)

    def _initialize_vars(self):
        self.hidden_p = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
        self.visible_recon_p = tf.nn.sigmoid(
            tf.matmul(sample_prob(self.hidden_p), tf.transpose(self.w)) + self.visible_bias
        )
        self.hidden_recon_p = tf.nn.sigmoid(tf.matmul(self.visible_recon_p, self.w) + self.hidden_bias)

        self.positive_grad = tf.matmul(tf.transpose(self.x), self.hidden_p)
        self.negative_grad = tf.matmul(tf.transpose(self.visible_recon_p), self.hidden_recon_p)

        f = lambda x: self.momentum * x + self.learning_rate * x * (1 - self.momentum)

        delta_w = f(self.positive_grad - self.negative_grad)
        delta_visible_bias = f(tf.reduce_mean(self.x - self.visible_recon_p, 0))
        delta_hidden_bias = f(tf.reduce_mean(self.hidden_p - self.hidden_recon_p, 0))

        self.update_w = self.w.assign(self.w + delta_w)
        self.update_visible_bias = self.visible_bias.assign(self.visible_bias + delta_visible_bias)
        self.update_hidden_bias = self.hidden_bias.assign(self.hidden_bias + delta_hidden_bias)
