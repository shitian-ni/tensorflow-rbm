import tensorflow as tf
import numpy as np
from .util import np_xavier_init


class RBM:
    def __init__(self, n_visible, n_hidden, *, learning_rate, momentum):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.x = tf.placeholder(tf.float32, [None, self.n_visible])
        self.hidden_placeholder = tf.placeholder(tf.float32, [None, self.n_hidden])
        self.w = tf.placeholder(tf.float32, [self.n_visible, self.n_hidden])
        self.visible_bias = tf.placeholder(tf.float32, [self.n_visible])
        self.hidden_bias = tf.placeholder(tf.float32, [self.n_hidden])

        self.visible_p = None
        self.update_w = None
        self.update_visible_bias = None
        self.update_hidden_bias = None

        self._initialize_vars()

        self.compute_hidden = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
        self.compute_visible = tf.nn.sigmoid(tf.matmul(self.compute_hidden, tf.transpose(self.w)) + self.visible_bias)
        self.compute_err = tf.reduce_mean(tf.square(self.x - self.visible_p))

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_vars(self):
        pass

    def compute_err(self, batch_x):
        return self.sess.run(self.compute_err, feed_dict={self.x: batch_x})

    def compute_free_energy(self):
        pass

    def transform(self, batch_x):
        return self.sess.run(self.compute_hidden, feed_dict={self.x: batch_x})

    def reconstruct(self, batch_x):
        return self.sess.run(self.compute_visible, feed_dict={self.x: batch_x})

    def partial_fit(self, batch_x):
        self.sess.run([self.update_w, self.update_visible_bias, self.update_hidden_bias],
                      feed_dict={self.x: batch_x})

        return self.sess.run(self.compute_err, feed_dict={self.x: batch_x})

    def get_weights(self):
        return self.w.eval(self.sess), \
               self.visible_bias.eval(self.sess),\
               self.hidden_bias.eval(self.sess)

    def save_weights(self, path, weights_names):
        saver = tf.train.Saver({weights_names[0]: self.w,
                                weights_names[1]: self.visible_bias,
                                weights_names[2]: self.hidden_bias})
        return saver.save(self.sess, path)

    def set_weights(self, w, visible_bias, hidden_bias):
        self.sess.run(self.w.assign(w))
        self.sess.run(self.visible_bias.assign(visible_bias))
        self.sess.run(self.hidden_bias.assign(hidden_bias))

    def load_weights(self, path, weights_names):
        saver = tf.train.Saver({weights_names[0]: self.w,
                                weights_names[1]: self.visible_bias,
                                weights_names[2]: self.hidden_bias})
        saver.restore(self.sess, path)

