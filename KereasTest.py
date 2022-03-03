import numpy as np
np.random.seed(123)
# Git - trial
# import keras as ke

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPool2D
from keras.utils import np_utils

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
