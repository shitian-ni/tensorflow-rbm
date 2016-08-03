import numpy as np
import tensorflow as tf


def tf_xavier_init(fan_in, fan_out, *, const=1.0, dtype=np.float32):
    low = -const * np.sqrt(6.0 / (fan_in + fan_out))
    high = const * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=dtype)


def np_xavier_init(fan_in, fan_out, *, const=1.0, dtype=np.float32):
    low = -const * np.sqrt(6.0 / (fan_in + fan_out))
    high = const * np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(size=(fan_in, fan_out), low=low, high=high).astype(dtype)


def sample_prob(probs):
    return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))