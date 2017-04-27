import os
# supress tensorflow logging other than errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from attacks.fgsm import fgsm


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
y_train = np_utils.to_categorical(y_train, nb_classes)
z0 = y_test.copy()
y_test = np_utils.to_categorical(y_test, nb_classes)


sess = tf.InteractiveSession()
K.set_session(sess)


if False:
    print('\nLoading model')
    model = load_model('model/ex_00.h5')
else:
    print('\nBuilding model')
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), padding='same',
               input_shape=input_shape),
        Activation('relu'),
        Conv2D(filters=32, kernel_size=(3, 3), padding='same'),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2), padding='same'),
        Flatten(),
        Dense(128),
        Activation('relu'),
        Dropout(0.5),
        Dense(10),
        Activation('softmax')])

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print('\nTraining model')
    model.fit(X_train, y_train, epochs=5)

    print('\nSaving model')
    os.makedirs('model', exist_ok=True)
    model.save('model/ex_00.h5')


x = tf.placeholder(tf.float32, (None, img_rows, img_cols, img_chas))
y = tf.placeholder(tf.float32, (None, nb_classes))

def _model_fn(x, logits=False):
    ybar = model(x)
    logits_, = ybar.op.inputs
    if logits:
        return ybar, logits_
    return ybar

x_adv = fgsm(_model_fn, x, epochs=9, eps=0.02)


print('\nTest against clean data')
score = model.evaluate(X_test, y_test)
print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))


nb_sample = X_test.shape[0]
batch_size = 128
nb_batch = int(np.ceil(nb_sample/batch_size))
X_adv = np.empty(X_test.shape)
for batch in range(nb_batch):
    print('batch {0}/{1}'.format(batch+1, nb_batch), end='\r')
    start = batch * batch_size
    end = min(nb_sample, start+batch_size)
    tmp = sess.run(x_adv, feed_dict={x: X_test[start:end],
                                     y: y_test[start:end],
                                     K.learning_phase(): 0})
    X_adv[start:end] = tmp


print('\nTest against adversarial data')
score = model.evaluate(X_adv, y_test)
print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))


if False:
    db = np.load('data/ex_00.npy')
    X_tmp, y_adv = db['X_tmp'], db['y_adv']
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
    with open('data/ex_00.npy', 'wb') as w:
        np.savez(w, X_tmp=X_tmp, y_adv=y_adv)


print('\nPlotting results')
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

print('\nSaving figure')
gs.tight_layout(fig)
os.makedirs('img', exist_ok=True)
plt.savefig('img/ex_00.png')
