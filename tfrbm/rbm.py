import tensorflow as tf
import numpy as np
from .util import tf_xavier_init


class RBM:
    def __init__(self, n_visible, n_hidden, learning_rate=1.0, momentum=1.0, xavier_const=1.0, err_function='mse'):
        assert 0.0 <= momentum <= 1.0
        assert err_function == 'mse' or err_function == 'cosine'

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.x = tf.placeholder(tf.float32, [None, self.n_visible])
        self.y = tf.placeholder(tf.float32, [None, self.n_hidden])

        self.w = tf.Variable(tf_xavier_init(self.n_visible, self.n_hidden, const=xavier_const), dtype=tf.float32)
        self.visible_bias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.hidden_bias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        self.delta_w = tf.Variable(tf.zeros([self.n_visible, self.n_hidden]), dtype=tf.float32)
        self.delta_visible_bias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.delta_hidden_bias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        self.update_weights = None
        self.update_deltas = None
        self.compute_hidden = None
        self.compute_visible = None
        self.compute_visible_from_hidden = None

        self._initialize_vars()

        assert self.update_weights is not None
        assert self.update_deltas is not None
        assert self.compute_hidden is not None
        assert self.compute_visible is not None
        assert self.compute_visible_from_hidden is not None

        if err_function == 'cosine':
            x1_norm = tf.nn.l2_normalize(self.x, 1)
            x2_norm = tf.nn.l2_normalize(self.compute_visible, 1)
            cos_val = tf.reduce_mean(tf.reduce_sum(tf.mul(x1_norm, x2_norm), 1))
            self.compute_err = tf.acos(cos_val) / tf.constant(np.pi)
        else:
            self.compute_err = tf.reduce_mean(tf.square(self.x - self.compute_visible))

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_vars(self):
        pass

    def get_err(self, batch_x):
        return self.sess.run(self.compute_err, feed_dict={self.x: batch_x})

    def get_free_energy(self):
        pass

    def transform(self, batch_x):
        return self.sess.run(self.compute_hidden, feed_dict={self.x: batch_x})

    def transform_inv(self, batch_y):
        return self.sess.run(self.compute_visible_from_hidden, feed_dict={self.y: batch_y})

    def reconstruct(self, batch_x):
        return self.sess.run(self.compute_visible, feed_dict={self.x: batch_x})

    def partial_fit(self, batch_x):
        self.sess.run(self.update_weights + self.update_deltas, feed_dict={self.x: batch_x})

    def fit(self, data_x, n_epoches=10, batch_size=10, shuffle=True, verbose=True, tqdm=None):
        n_data = data_x.shape[0]
        n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)

        assert n_batches > 0

        if shuffle:
            data_x_cpy = data_x.copy()
        else:
            data_x_cpy = data_x

        if shuffle:
            inds = np.arange(n_data)

        if tqdm:
            if tqdm == 'notebook':
                from tqdm import tqdm_notebook as tq
            else:
                from tqdm import tqdm as tq

        epoches_range = range(n_epoches)

        if tqdm:
            epoches_range = tq(epoches_range, desc='Epoch')

        errs = []

        for e in epoches_range:
            epoch_errs = np.zeros((n_batches,))
            epoch_errs_ptr = 0

            if shuffle:
                np.random.shuffle(inds)
                data_x_cpy = data_x_cpy[inds]

            batches_range = range(n_batches)

            if tqdm:
                batches_range = tq(batches_range, desc='Batch')

            for b in batches_range:
                batch_x = data_x_cpy[b * batch_size:(b + 1) * batch_size]
                self.partial_fit(batch_x)
                batch_err = self.get_err(batch_x)
                epoch_errs[epoch_errs_ptr] = batch_err
                epoch_errs_ptr += 1

            if verbose:
                err_mean = epoch_errs.mean()
                print('Epoch: {:d}, error: {:4f}'.format(e, err_mean))

            errs = np.hstack([errs, epoch_errs])

        return errs

    def get_weights(self):
        return self.sess.run(self.w),\
            self.sess.run(self.visible_bias),\
            self.sess.run(self.hidden_bias)

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
