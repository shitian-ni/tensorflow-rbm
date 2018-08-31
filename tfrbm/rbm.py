from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
from .util import tf_xavier_init


class RBM:
    def __init__(self,
                 n_visible,
                 n_hidden,
                 learning_rate=5,
                 momentum=0,
                 xavier_const=1.0,
                 err_function='mse',
                 use_tqdm=False,
                 # DEPRECATED:
                 tqdm=None):
        if not 0.0 <= momentum <= 1.0:
            raise ValueError('momentum should be in range [0, 1]')

        if err_function not in {'mse', 'cosine'}:
            raise ValueError('err_function should be either \'mse\' or \'cosine\'')

        self._use_tqdm = use_tqdm
        self._tqdm = None

        if use_tqdm or tqdm is not None:
            from tqdm import tqdm
            self._tqdm = tqdm

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

            

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_vars(self):
        pass

    def get_err(self, batch_x, verbose = False):
        if verbose:
            print("x: ",self.sess.run(self.x, feed_dict={self.x: batch_x}), 
                "compute_visible: ",self.sess.run(self.compute_visible, feed_dict={self.x: batch_x}))
        return self.sess.run(self.compute_err, feed_dict={self.x: batch_x})

    def get_energy(self, data):
        return self.sess.run(self.compute_energy, feed_dict={self.x: data})

    def transform(self, batch_x):
        return self.sess.run(self.compute_hidden, feed_dict={self.x: batch_x})

    def transform_inv(self, batch_y):
        return self.sess.run(self.compute_visible_from_hidden, feed_dict={self.y: batch_y})

    def reconstruct(self, batch_x):
        return self.sess.run(self.compute_visible, feed_dict={self.x: batch_x})

    def partial_fit(self, batch_x):
        self.sess.run(self.update_weights + self.update_deltas, feed_dict={self.x: batch_x})

    def fit(self,
            data_x,
            n_epoches=10,
            batch_size=10,
            learning_rate=5,
            decay = 0,
            shuffle=True,
            verbose=True, 
            epochs_to_test = 1,
            early_stop = True):
        assert n_epoches > 0

        self.learning_rate = learning_rate

        n_data = data_x.shape[0]

        if batch_size > 0:
            n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
        else:
            n_batches = 1

        if shuffle:
            data_x_cpy = data_x.copy()
            inds = np.arange(n_data)
        else:
            data_x_cpy = data_x

        errs = []

        delta_energies = []

        sample = self.reconstruct(np.random.rand(1,self.n_visible))[0]>=0.5

        if hasattr(self, 'image_height'):
            plt.figure()
            plt.axis('off')
            plt.title("Image reconstructed before training ", y=1.03)
                
            plt.imshow(sample.reshape(self.image_height, -1))


        for e in range(n_epoches):
            # if verbose and e % 100 == 0 and not self._use_tqdm:
            #     print('Epoch: {:d}'.format(e))

            epoch_errs = np.zeros((n_batches,))
            epoch_errs_ptr = 0

            if shuffle:
                np.random.shuffle(inds)
                data_x_cpy = data_x_cpy[inds]

            r_batches = range(n_batches)

            if verbose and self._use_tqdm:
                r_batches = self._tqdm(r_batches, desc='Epoch: {:d}'.format(e), ascii=True, file=sys.stdout)

            for b in r_batches:
                batch_x = data_x_cpy[b * batch_size:(b + 1) * batch_size]
                self.partial_fit(batch_x)
                batch_err = self.get_err(batch_x)
                epoch_errs[epoch_errs_ptr] = batch_err
                epoch_errs_ptr += 1

            if verbose and e % 1000 == 0:
                err_mean = epoch_errs.mean()
                # if self._use_tqdm:
                #     self._tqdm.write('Train error: {:.4f}'.format(err_mean))
                #     self._tqdm.write('')
                # elif e < 5000:
                #     print('Epoch: {:d}'.format(e),'Train error: {:.4f}'.format(err_mean))
                    
                sys.stdout.flush()

            errs = np.hstack([errs, epoch_errs])

            self.learning_rate *= (1-decay)

            sample = self.reconstruct(np.random.rand(1,self.n_visible))[0]>=0.5

            original = data_x

            original = data_x[0]
            samples = sample[:self.n_visible]
            if type(original) != list:
                original = original.tolist()
            if type(samples) != list:
                samples = samples.tolist()

            def show_img(self, e):
                    
                plt.figure()
                plt.axis('off')
                plt.title("Image reconstructed after training "+str(e+1)+" epochs", y=1.03)
                    
                plt.imshow(sample.reshape(self.image_height, -1))

            if early_stop and original == samples:
                print ("Stopped training early because the model can reconstruct the inputs")
                if hasattr(self, 'image_height'):
                    show_img(self, e)
                break
            if e % epochs_to_test == 0:
                if hasattr(self, 'image_height'):
                    show_img(self, e)

            if e%20000 == 0:
                pass
                # print("------Epoch: {:d}   ------")
                # prediction, energy = self.predict([[0,0,-1]],[2])
                # delta_energy = energy[0]-energy[1]
                # delta_energies.append(delta_energy)
                # print("Predicting most difficult to train \"0 xor 0 = ".format(e),
                #     prediction," delta_energy = ",delta_energy,"\n------------")

        return errs

    def predict(self, data, positions_to_predict):
        data = np.array(data)
        min_energy = 10000000
        best_answer = []
        need_to_predict = len(positions_to_predict)
        total_possibilities_num = 2**need_to_predict

        data = np.repeat(data,total_possibilities_num,axis=0)

        for possibility_idx, possibles in enumerate(range(total_possibilities_num)):
            for idx,possible in enumerate(bin(possibles)[2:]):
                data[possibility_idx, positions_to_predict[idx]]=int(possible)
        # print("All possibilities: ",data)
        energy = self.get_energy(data)
        # print("energy:",energy)
        best_answer_index = np.argmin(energy)
        # print("best_answer_index:",best_answer_index)
        min_energy = energy[best_answer_index]
        best_answer = data[best_answer_index]
       
        return best_answer[positions_to_predict], energy

    def get_weights(self):
        return self.sess.run(self.w),\
            self.sess.run(self.visible_bias),\
            self.sess.run(self.hidden_bias)

    def save_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w': self.w,
                                name + '_v': self.visible_bias,
                                name + '_h': self.hidden_bias})
        return saver.save(self.sess, filename)

    def set_weights(self, w, visible_bias, hidden_bias):
        self.sess.run(self.w.assign(w))
        self.sess.run(self.visible_bias.assign(visible_bias))
        self.sess.run(self.hidden_bias.assign(hidden_bias))

    def load_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w': self.w,
                                name + '_v': self.visible_bias,
                                name + '_h': self.hidden_bias})
        saver.restore(self.sess, filename)
