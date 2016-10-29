# tensorflow-rbm

Tensorflow implementation of Restricted Boltzman Machine for layerwise pretraining of deep autoencoders.

### Overview

This is a fork of https://github.com/Cospel/rbm-ae-tf with some corrections and improvements:

- scripts are in the package now
- implemented momentum for RBM
- using probabilities instead of samples for training
- implemented both Bernoulli-Bernoulli RBM and Gaussian-Bernoulli RBM

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
