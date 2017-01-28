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

from attacks.jsma import jsma


print('Loading mnist')
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

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
    model.save('model/ex_01.h5')
else:
    model = load_model('model/ex_01.h5')


x = tf.placeholder(tf.float32, (None, 28, 28, 1))
y = tf.placeholder(tf.float32, (None, 10))
ybar = model(x)
target = tf.placeholder(tf.int32, ())
x_adv = jsma(model, x, target)


print('Testing...')
score = model.evaluate(X_test, y_test)
print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))


saved = True

if saved:
    with gzip.open('data/ex_01.pkl.gz', 'rb') as r:
        X_adv = pickle.load(r)
        X_adv = np.array(X_adv)
else:
    thres = 0.9
    y_pred = model.predict(X_test)
    y0 = np.max(y_pred, axis=1)
    z0 = np.argmax(y_pred, axis=1)
    ind = y0 > thres
    x0 = X_test[ind]
    z0 = z0[ind]

    X_adv = np.empty((10, 10, 28, 28))
    for i in np.arange(10):
        print('Source label {0}'.format(i))
        ind = z0 == i
        X_i_all = x0[ind]
        while True:
            found = True
            cur = np.random.choice(X_i_all.shape[0])
            X_i = X_i_all[cur, np.newaxis]
            for j in np.arange(10):
                print(' Target label {0}'.format(j), end='')
                if j == i:
                    X_i_adv = X_i.copy()
                else:
                    X_i_adv = sess.run(x_adv, feed_dict={
                        x: X_i, target: j, K.learning_phase(): 0})
                y_i_adv = model.predict(X_i_adv)
                y1 = np.max(y_i_adv)
                z1 = np.argmax(y_i_adv)
                found = z1==j
                if not found:
                    print(' Fail')
                    break
                X_adv[i, j] = np.squeeze(X_i_adv)
                print(' res: {0} {1:.2f}'.format(z1==j, y1))

            if found:
                break

    os.makedirs('data', exist_ok=True)
    with gzip.open('data/ex_01.pkl.gz', 'wb') as w:
        pickle.dump(X_adv.tolist(), w)


# Generaing figures
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(10, 10, wspace=0.1, hspace=0.1)

for i in range(10):
    for j in range(10):
        ax = fig.add_subplot(gs[i, j])
        ax.imshow(X_adv[i, j], cmap='gray', interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])

        if i == j:
            for spine in ax.spines:
                ax.spines[spine].set_color('green')
                ax.spines[spine].set_linewidth(5)

        if ax.is_first_col():
            ax.set_ylabel(i, fontsize=20, rotation='horizontal',
                          ha='right')
        if ax.is_last_row():
            ax.set_xlabel(j, fontsize=20)

gs.tight_layout(fig)
os.makedirs('img', exist_ok=True)
plt.savefig('img/ex_01.png')
