import os
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

from attacks.jsma import jsma2


print('Loading mnist')
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# assume K.image_dim_ordering() == 'tf'
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
input_shape = (28, 28, 1)

# one hot encoding
z_train = y_train.copy()
y_train = np_utils.to_categorical(y_train, 10)
z_test = y_test.copy()
y_test = np_utils.to_categorical(y_test, 10)

batch_size = 64
nb_epoch = 10
saved = True

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

    model.fit(X_train, y_train, nb_epoch=nb_epoch)

    os.makedirs('model', exist_ok=True)
    model.save('model/ex_04.h5')
else:
    model = load_model('model/ex_04.h5')


x = tf.placeholder(tf.float32, (None, 28, 28, 1))
y = tf.placeholder(tf.float32, (None, 10))
ybar = model(x)
target = tf.placeholder(tf.int32, ())
x_adv = jsma2(model, x, target, delta=1.)


print('Testing...')
score = model.evaluate(X_test, y_test)
print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))


saved = False

if saved:
    with gzip.open('data/ex_04.pkl.gz', 'rb') as r:
        X_adv, p = pickle.load(r)
        X_adv, p = np.array(X_adv), np.array(p)
else:
    thres = 0.9
    blank = np.zeros((1, 28, 28, 1))
    X_adv = np.empty((10, 28, 28))
    p = np.empty((10,))
    for i in np.arange(10):
        maxiter = 30
        print('Target label {0}'.format(i), end='')
        X_i_adv = sess.run(x_adv, feed_dict={
            x: blank, target: i, K.learning_phase(): 0})
        y_i_adv = model.predict(X_i_adv)
        y1 = np.max(y_i_adv)
        z1 = np.argmax(y_i_adv)
        X_adv[i] = np.squeeze(X_i_adv)
        p[i] = y1
        print(' proba: {0:.2f}'.format(y1))

    os.makedirs('data', exist_ok=True)
    with gzip.open('data/ex_04.pkl.gz', 'wb') as w:
        pickle.dump([X_adv.tolist(), p.tolist()], w)


# Generaing figures
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
plt.savefig('img/ex_04.png')
