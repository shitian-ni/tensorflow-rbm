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
errs = bbrbm.fit(mnist_images, n_epoches=20, batch_size=128, tqdm='notebook')
plt.plot(errs)
plt.show()
```

Output:
```
Epoch: 0, error: 0.079566
Epoch: 1, error: 0.047970
Epoch: 2, error: 0.040229
Epoch: 3, error: 0.036163
Epoch: 4, error: 0.033523
Epoch: 5, error: 0.031571
Epoch: 6, error: 0.030071
Epoch: 7, error: 0.028901
Epoch: 8, error: 0.027961
Epoch: 9, error: 0.027207
```

![Error plot](https://habrastorage.org/files/861/c6d/f0d/861c6df0d1604e49b4a710d7f0828cbf.png)

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

![3 reconstructed](https://habrastorage.org/files/eb3/d8b/8c8/eb3d8b8c8ddb48a384f4ddd43a5ef155.png)

![4 original](https://habrastorage.org/files/1c8/2ba/0e9/1c82ba0e906f4cb49ec6fa0e2e5bfafe.png)

![4 reconstructed](https://habrastorage.org/files/cbb/e65/5ff/cbbe655ff66049348418991b8084088a.png)

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
