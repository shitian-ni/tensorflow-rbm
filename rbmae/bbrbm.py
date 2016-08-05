import tensorflow as tf
from .rbm import RBM
from .util import sample_prob


class BBRBM(RBM):
    def __init__(self, *args, **kwargs):
        RBM.__init__(self, *args, **kwargs)

    def _initialize_vars(self):
        self.hidden_p = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
        self.visible_recon_p = tf.nn.sigmoid(tf.matmul(sample_prob(self.hidden_p), tf.transpose(self.w)) + self.visible_bias)
        self.hidden_recon_p = tf.nn.sigmoid(tf.matmul(self.visible_recon_p, self.w) + self.hidden_bias)

        self.positive_grad = tf.matmul(tf.transpose(self.x), self.hidden_p)
        self.negative_grad = tf.matmul(tf.transpose(self.visible_recon_p), self.hidden_recon_p)

        c = self.momentum + self.learning_rate * (1 - self.momentum)
        f = lambda x: c * x

        delta_w = f(self.positive_grad - self.negative_grad)
        delta_visible_bias = f(tf.reduce_mean(self.x - self.visible_recon_p, 0))
        delta_hidden_bias = f(tf.reduce_mean(self.hidden_p - self.hidden_recon_p, 0))

        self.update_w = self.w.assign(self.w + delta_w)
        self.update_visible_bias = self.visible_bias.assign(self.visible_bias + delta_visible_bias)
        self.update_hidden_bias = self.hidden_bias.assign(self.hidden_bias + delta_hidden_bias)

        self.compute_hidden = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
        self.compute_visible = tf.nn.sigmoid(tf.matmul(self.compute_hidden, tf.transpose(self.w)) + self.visible_bias)
        self.compute_visible_from_hidden = tf.nn.sigmoid(tf.matmul(self.y, tf.transpose(self.w)) + self.visible_bias)
