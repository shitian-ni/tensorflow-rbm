import tensorflow as tf
import numpy as np
from .rbm import RBM
from .util import sample_prob


class GBRBM(RBM):
    def __init__(self, *args, **kwargs):
        RBM.__init__(self, *args, **kwargs)

    def _initialize_vars(self):
        self.hidden_p = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
        self.visible_recon_p = tf.matmul(sample_prob(self.hidden_p), tf.transpose(self.w)) + self.visible_bias
        self.hidden_recon_p = tf.nn.sigmoid(tf.matmul(self.visible_recon_p, self.w) + self.hidden_bias)

        self.positive_grad = tf.matmul(tf.transpose(self.x), self.hidden_p)
        self.negative_grad = tf.matmul(tf.transpose(self.visible_recon_p), self.hidden_recon_p)

        f = lambda x_old, x_new: self.momentum * x_old + self.learning_rate * x_new * (1 - self.momentum)

        self.delta_w_old_store = np.zeros((self.n_visible, self.n_hidden), dtype=np.float32)
        self.delta_visible_bias_old_store = np.zeros((self.n_visible,), dtype=np.float32)
        self.delta_hidden_bias_old_store = np.zeros((self.n_hidden,), dtype=np.float32)

        self.delta_w_old = tf.placeholder(tf.float32, [self.n_visible, self.n_hidden])
        self.delta_visible_bias_old = tf.placeholder(tf.float32, [self.n_visible])
        self.delta_hidden_bias_old = tf.placeholder(tf.float32, [self.n_hidden])

        self.delta_w = f(self.delta_w_old, self.positive_grad - self.negative_grad)
        self.delta_visible_bias = f(self.delta_visible_bias_old, tf.reduce_mean(self.x - self.visible_recon_p, 0))
        self.delta_hidden_bias = f(self.delta_hidden_bias_old, tf.reduce_mean(self.hidden_p - self.hidden_recon_p, 0))

        self.update_w = self.w.assign(self.w + self.delta_w)
        self.update_visible_bias = self.visible_bias.assign(self.visible_bias + self.delta_visible_bias)
        self.update_hidden_bias = self.hidden_bias.assign(self.hidden_bias + self.delta_hidden_bias)

        self.compute_hidden = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
        self.compute_visible = tf.matmul(self.compute_hidden, tf.transpose(self.w)) + self.visible_bias
        self.compute_visible_from_hidden = tf.matmul(self.y, tf.transpose(self.w)) + self.visible_bias

