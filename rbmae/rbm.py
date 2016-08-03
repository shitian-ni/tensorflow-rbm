import tensorflow as tf
import numpy as np
from .util import np_xavier_init


class RBM:
    def __init__(self, n_visible, n_hidden, *, learning_rate=0.1, momentum=0.5):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.alpha = learning_rate
        self.momentum = momentum

        self._initialize_weights()

        self.x = tf.placeholder(tf.float32, [None, self.n_visible])
        self.w = tf.placeholder(tf.float32, [self.n_visible, self.n_hidden])
        self.visible_bias = tf.placeholder(tf.float32, [self.n_visible])
        self.hidden_bias = tf.placeholder(tf.float32, [self.n_hidden])

    def _initialize_weights(self):
        self.curr_w = np_xavier_init(self.n_visible, self.n_hidden, const=1.0, dtype=np.float32)
        self.curr_visible_bias = np.zeros((self.n_visible,), dtype=tf.float32)
        self.curr_hidden_bias = np.zeros((self.n_hidden,), dtype=tf.float32)

    def compute_err(self):
        pass

    def compute_free_energy(self):
        pass

    def transform(self):
        pass

    def reconstruct(self):
        pass

    def partial_fit(self):
        pass

    def get_weights(self):
        pass

    def get_weights_np(self):
        pass

    def save_weights(self):
        pass

    def set_weights(self):
        pass

    def load_weights(self):
        pass