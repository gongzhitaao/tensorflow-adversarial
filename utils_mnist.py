import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Convolution2D, Flatten
from keras.utils import np_utils


def load_mnist(flatten=True, validation_split=None):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.

    if flatten:
        X_train = np.reshape(X_train, (-1, 784))
        X_test = np.reshape(X_test, (-1, 784))

    y_train = np.reshape(y_train, (-1, 1))
    y_test = np.reshape(y_test, (-1, 1))

    # one hot encoding
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    if validation_split is not None:
        n = X_train.shape[0]
        nb_samples = int(n * (1-validation_split))
        X_train, X_val = X_train[:nb_samples], X_train[nb_samples:]
        y_train, y_val = y_train[:nb_samples], y_train[nb_samples:]
        return X_train, y_train, X_test, y_test, X_val, y_val

    return X_train, y_train, X_test, y_test


def model_mnist_mlp():
    model = Sequential()
    model.add(Dense(100, input_dim=784))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model
