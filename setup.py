from distutils.core import setup

desc = """\
RBMAE
==============
Tensorflow implementation of Restricted Boltzman Machine and Autoencoder for layerwise pretraining of Deep Autoencoders with RBM.
"""

setup(name='rbmae',
      version='0.0.1',
      author='Michal Lukac, Egor Malykh',
      author_email='fnk@fea.st',
      long_description=desc,
      packages=['rbmae'],
      url='https://github.com/meownoid/rbmae')