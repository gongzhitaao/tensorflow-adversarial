import os
# supress tensorflow logging other than errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from attacks.tgsm import tgsm


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
y_train = np_utils.to_categorical(y_train, nb_classes)
z_test = y_test.copy()
y_test = np_utils.to_categorical(y_test, nb_classes)


sess = tf.InteractiveSession()
K.set_session(sess)


if False:
    print('\nLoading model')
    model = load_model('model/ex_02.h5')
else:
    print('\nBuilding model')
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        Activation('relu'),
        Conv2D(32, (3, 3), padding='same'),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2), padding='same'),
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
    model.fit(X_train, y_train, epochs=10)

    print('\nSaving model')
    os.makedirs('model', exist_ok=True)
    model.save('model/ex_02.h5')


x = tf.placeholder(tf.float32, (None, img_rows, img_cols, img_chas))
y = tf.placeholder(tf.int32, ())
x_adv = tgsm(model, x, y, eps=0.01, epochs=30)


print('\nTest against clean data')
score = model.evaluate(X_test, y_test)
print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))


if False:
    X_adv = np.load('data/ex_02.npy')
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
        for cur in range(X_i_all.shape[0]):
            found = True
            X_i = X_i_all[cur, np.newaxis]
            for j in np.arange(10):
                print(' [{0}/{1}] {2} --> {3}'
                      .format(cur, X_i_all.shape[0], i, j), end='')
                if j == i:
                    X_i_adv = X_i.copy()
                else:
                    X_i_adv = sess.run(x_adv, feed_dict={
                        x: X_i, y: j, K.learning_phase(): 0})
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
    np.save('data/ex_02.npy', X_adv)


print('\nGenerating figure')
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
plt.savefig('img/ex_02.png')
