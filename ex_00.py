import os
import time
import gzip
import pickle

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from attacks.fgsm import fgsm


print('Loading mnist')
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# assume K.image_dim_ordering() == 'tf'
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
input_shape = (28, 28, 1)

# one hot encoding
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

batch_size = 64
nb_epoch = 10
saved = False

sess = tf.InteractiveSession()
K.set_session(sess)

print('Building model')
if not saved:
    model = Sequential()
    model.add(Convolution2D(32, 3, 3,
                            border_mode='valid',
                            input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    ybar = model(x)
    x_adv = fgsm(x, ybar, eps=0.3)

    model.fit(X_train, y_train, nb_epoch=nb_epoch)

    os.makedirs('model', exist_ok=True)
    model.save('model/ex_00.h5')
else:
    model = load_model('model/ex_00.h5')

print('Testing...')
score = model.evaluate(X_test, y_test)
print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))


nb_sample = X_test.shape[0]
nb_batch = int(np.ceil(nb_sample/batch_size))
X_adv = np.empty(X_test.shape)
with timer('Craft adversarial images'):
    for batch in range(nb_batch):
        print('batch {0}/{1}'.format(batch+1, nb_batch), end='\r')
        start = batch * batch_size
        end = min(nb_sample, start+batch_size)
        tmp = sess.run(x_adv, feed_dict={x: X_test[start:end],
                                         K.learning_phase(): 0})
        X_adv[start:end] = tmp
print('Elapsed {0:.2f}s'.format(tock-tick))


print('Testing against adversarial test data')
score = model.evaluate(X_adv, y_test)
print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))
