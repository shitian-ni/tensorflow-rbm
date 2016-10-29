# tensorflow-rbm

Tensorflow implementation of Restricted Boltzman Machine for layerwise pretraining of deep autoencoders.

### Overview

This is a fork of https://github.com/Cospel/rbm-ae-tf with some corrections and improvements:

- scripts are in the package now
- implemented momentum for RBM
- using probabilities instead of samples for training
- implemented both Bernoulli-Bernoulli RBM and Gaussian-Bernoulli RBM

### BBRBM Example
Bernoulli-Bernoulli RBM is good for Bernoulli-distributed binary input data. MNIST, for example.

Load data and train RBM:
```python
import numpy as np
import matplotlib.pyplot as plt
from tfrbm import BBRBM, GBRBM
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist_images = mnist.train.images

bbrbm = BBRBM(n_visible=784, n_hidden=64, learning_rate=0.1, momentum=0.95)
errs = bbrbm.fit(mnist_images, n_epoches=20, batch_size=10, tqdm='notebook')
plt.plot(errs)
plt.show()
```

Output:
```
Epoch: 0, error: 0.069226
Epoch: 1, error: 0.042563
Epoch: 2, error: 0.036503
Epoch: 3, error: 0.033372
Epoch: 4, error: 0.031310
Epoch: 5, error: 0.029698
...
Epoch: 17, error: 0.022318
Epoch: 18, error: 0.022065
Epoch: 19, error: 0.021828


```

![Error plot](https://habrastorage.org/files/804/985/f56/804985f56399412b8fab7cae1439cfda.png)

Examine some reconstructed data:
```python
IMAGE = 1

def show_digit(x):
    plt.imshow(x.reshape((28, 28)), cmap=plt.cm.gray)
    plt.show()

image = mnist_images[IMAGE]
show_digit(image)
image_rec = bbrbm.reconstruct(image.reshape(1,-1))
show_digit(image_rec)
```

Examples:

![3 original](https://habrastorage.org/files/fa2/a3e/35b/fa2a3e35b2cd417fa70de3e6aa146464.png)

![3 reconstructed](https://habrastorage.org/files/45b/00b/b89/45b00bb891c04582adef436af7501ffc.png)

![4 original](https://habrastorage.org/files/1c8/2ba/0e9/1c82ba0e906f4cb49ec6fa0e2e5bfafe.png)

![4 reconstructed](https://habrastorage.org/files/0c0/c17/4c6/0c0c174c638847d397067a9dc504902b.png)

### API

```python
rbm = BBRBM(n_visible, n_hidden, learning_rate=1.0, momentum=1.0, xavier_const=1.0, err_function='mse')
```
or
```python
rbm = GBRBM(n_visible, n_hidden, learning_rate=1.0, momentum=1.0, xavier_const=1.0, err_function='mse', sample_visible=False, sigma=1)
```

Initialization.

* `n_visible` — number of neurons on visible layer
* `n_hidden` — number of neurons on hidden layer
* `xavier_const` — constant, used in weights initialization, 1.0 is good
* `err_function` — error function, it's NOT USED in train process, just in `get_err` function, should be `mse` or `cosine`

Only for `GBRBM`:

* `sample_visible` — sample reconstructed data with Gaussian distribution (reconstructed data as mean and `sigma` parameter as deviation) or not (if not, every gaussoid will be projected into one point)
* `sigma` — standard deviation of the input data

Advices:

* Use BBRBM for Bernoulli distributed data. Input values in this case shoud be in the interval from `0` to `1`.

```python
rbm.fit(data_x, n_epoches=10, batch_size=10, shuffle=True, verbose=True, tqdm=None)
```

Fit the model.

* `data_x` — data of shape `(n_data, data_dim)`
* `n_epoches` — number of epoches
* `batch_size` — batch size, should be as small as possible
* `shuffle` — shuffle data or not
* `verbose` — output to stdout
* `tqdm` — use tqdm package or not, should be None, True or 'notebook'

Returns errors vector.

```python
rbm.partial_fit(batch_x)
```

Fit the model on one batch.

```python
rbm.reconstruct(batch_x)
```

Reconstruct data. Input and output shapes are `(n_data, n_visible)`.

```python
rbm.transform(batch_x)
```

Transform data. Input shape is `(n_data, n_visible)`, output shape is `(n_data, n_hidden)`.

```python
rbm.transform_inv(batch_y)
```

Inverse transform data. Input shape is `(n_data, n_hidden)`, output shape is `(n_data, n_visible)`.

```python
rbm.get_err(batch_x)
```

Returns error on batch.

```python
rbm.get_weights()
```

Get RBM's weights as a numpy arrays. Returns `(W, Bv, Bh)` where `W` is weights matrix of shape `(n_visible, n_hidden)`, `Bv` is visible layer bias of shape `(n_visible,)` and `Bh` is hidden layer bias of shape `(n_hidden,)`.

```python
rbm.set_weights(w, visible_bias, hidden_bias)
```

Set RBM's weights as numpy arrays.

```python
rbm.save_weights(path, name)
```

Save RBM's weights to `filename` file with unique `name` prefix.

```python
rbm.load_weights(path, name)
```

Loads RBM's weights from `filename` file with unique `name` prefix.

### Original README

Tensorflow implementation of Restricted Boltzman Machine and Autoencoder for layerwise pretraining of Deep Autoencoders with RBM. Idea is to first create RBMs for pretraining weights for autoencoder. Then weigts for autoencoder are loaded and autoencoder is trained again. In this implementation you can also use tied weights for autoencoder(that means that encoding and decoding layers have same transposed weights!).

I was inspired with these implementations but I need to refactor them and improve them. I tried to use also similar api as it is in [tensorflow/models](https://github.com/tensorflow/models):

> [myme5261314](https://gist.github.com/myme5261314/005ceac0483fc5a581cc)

> [saliksyed](https://gist.github.com/saliksyed/593c950ba1a3b9dd08d5)

> Thank you for your gists!

More about pretraining of weights in this paper:
> [Reducing the Dimensionality of Data with Neural Networks](https://www.cs.toronto.edu/~hinton/science.pdf)

Feel free to make updates, repairs. You can enhance implementation with some tips from:
> [Practical Guide to training RBM](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)
