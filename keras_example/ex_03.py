import os

# supress tensorflow logging other than errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import gzip
import pickle

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.utils import np_utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from attacks.smda import smda


img_rows = 28
img_cols = 28
img_chas = 1
input_shape = (img_rows, img_cols, img_chas)
nb_classes = 10


print('\nLoading mnist')
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

X_train = X_train.reshape(-1, img_rows, img_cols, img_chas)
X_test = X_test.reshape(-1, img_rows, img_cols, img_chas)

# one hot encoding
z_train = y_train.copy()
y_train = np_utils.to_categorical(y_train, 10)
z_test = y_test.copy()
y_test = np_utils.to_categorical(y_test, 10)


sess = tf.InteractiveSession()
K.set_session(sess)


if False:
    print('\nLoading model')
    model = load_model('model/ex_03.h5')
else:
    print('\nBuilding model')
    model = Sequential([
        Convolution2D(32, 3, 3, input_shape=input_shape),
        Activation('relu'),
        Convolution2D(32, 3, 3),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        # Dropout(0.25),
        Flatten(),
        Dense(128),
        Activation('relu'),
        # Dropout(0.5),
        Dense(10),
        Activation('softmax')])

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print('\nTraining model')
    model.fit(X_train, y_train, nb_epoch=5)

    print('\nSaving model')
    os.makedirs('model', exist_ok=True)
    model.save('model/ex_03.h5')


x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                      img_chas))
y = tf.placeholder(tf.float32, shape=(None, nb_classes))
target = tf.placeholder(tf.int32, ())
x_adv = smda(model, x, target, epochs=0.1, min_proba=0.8)


print('\nTest against clean data')
score = model.evaluate(X_test, y_test)
print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))


if False:
    db = np.load('data/ex_03.npy')
    X_adv, p = db['X_adv'], db['p']
else:
    blank = np.zeros((1, 28, 28, 1))
    X_adv = np.empty((10, 28, 28))
    p = np.empty((10,))
    for i in np.arange(10):
        maxiter = 30
        print('Target label {0}'.format(i), end='', flush=True)
        X_i_adv = sess.run(x_adv, feed_dict={
            x: blank, target: i, K.learning_phase(): 0})
        y_i_adv = model.predict(X_i_adv)
        y1 = np.max(y_i_adv)
        z1 = np.argmax(y_i_adv)
        X_adv[i] = np.squeeze(X_i_adv)
        p[i] = y1
        print(' proba: {0:.2f}'.format(y1))

    os.makedirs('data', exist_ok=True)
    with open('data/ex_03.npy', 'wb') as w:
        np.savez(w, X_adv=X_adv, p=p)


print('\nGenerating figure')

fig = plt.figure(figsize=(10, 1.8))
gs = gridspec.GridSpec(1, 10, wspace=0.1, hspace=0.1)

for i in range(10):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(X_adv[i], cmap='gray', interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('{0} ({1:.2f})'.format(i, p[i]), fontsize=12)

gs.tight_layout(fig)
os.makedirs('img', exist_ok=True)
plt.savefig('img/ex_03.png')
