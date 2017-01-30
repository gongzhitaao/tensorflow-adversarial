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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from attacks.llcm import llcm


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
z0 = y_test.copy()
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
    model.save('model/ex_05.h5')
else:
    model = load_model('model/ex_05.h5')


x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
y = tf.placeholder(tf.float32, shape=(None, 10))
x_adv = llcm(model, x, nb_epoch=4, eps=0.1)


print('Testing...')
score = model.evaluate(X_test, y_test)
print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))


nb_sample = X_test.shape[0]
nb_batch = int(np.ceil(nb_sample/batch_size))
X_adv = np.empty(X_test.shape)
for batch in range(nb_batch):
    print('batch {0}/{1}'.format(batch+1, nb_batch), end='\r')
    start = batch * batch_size
    end = min(nb_sample, start+batch_size)
    tmp = sess.run(x_adv, feed_dict={x: X_test[start:end],
                                     K.learning_phase(): 0})
    X_adv[start:end] = tmp


print('Testing against adversarial test data')
score = model.evaluate(X_adv, y_test)
print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))


saved = False

if saved:
    with gzip.open('data/ex_05.pkl.gz', 'rb') as r:
        X_tmp, y_adv = pickle.load(r)
        X_tmp, y_adv = np.array(X_tmp), np.array(y)
else:
    y1 = model.predict(X_test)
    z1 = np.argmax(y1, axis=1)
    y2 = model.predict(X_adv)
    z2 = np.argmax(y2, axis=1)

    X_tmp = np.empty((10, 28, 28))
    y_adv = np.empty((10, 10))
    for i in range(10):
        print('Target {0}'.format(i))
        ind, = np.where(np.all([z0==i, z1==i, z2!=i], axis=0))
        cur = np.random.choice(ind.shape[0])
        cur = ind[cur]
        X_tmp[i] = np.squeeze(X_adv[cur])
        y_adv[i] = y2[cur]

    os.makedirs('data', exist_ok=True)
    with gzip.open('data/ex_05.pkl.gz', 'wb') as w:
        pickle.dump([X_tmp.tolist(), y_adv.tolist()], w)


fig = plt.figure(figsize=(10, 1.8))
gs = gridspec.GridSpec(1, 10, wspace=0.1, hspace=0.1)

label = np.argmax(y_adv, axis=1)
p = np.max(y_adv, axis=1)
for i in range(10):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(X_tmp[i], cmap='gray', interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('{0} ({1:.2f})'.format(label[i], p[i]), fontsize=12)

gs.tight_layout(fig)
os.makedirs('img', exist_ok=True)
plt.savefig('img/ex_05.png')
