import tensorflow as tf
from .util import tf_xavier_init


class RBM:
    def __init__(self, n_visible, n_hidden, learning_rate=1.0, momentum=1.0, xavier_const=1.0, err_function='mse'):
        assert 0.0 < momentum < 1.0
        assert err_function == 'mse' or err_function == 'cossim'
        
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

        assert self.update_weights != None
        assert self.update_deltas != None
        assert self.compute_hidden != None
        assert self.compute_visible != None
        assert self.compute_visible_from_hidden != None
        
        if err_function == 'cossim':
            norm = lambda x: tf.sqrt(tf.reduce_sum(tf.square(x), 1))
            self.compute_err = 1 - tf.reduce_mean(tf.truediv(
                tf.reduce_sum(tf.mul(self.x, self.compute_visible), 1),
                tf.mul(norm(self.x), norm(self.compute_visible))
                ))
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

    def get_weights(self):
        return self.w.eval(session=self.sess), \
               self.visible_bias.eval(session=self.sess),\
               self.hidden_bias.eval(session=self.sess)

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

