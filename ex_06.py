import os
import time
import gzip
import pickle

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.advanced_activations import ParametricSoftplus
from keras.callbacks import EarlyStopping
from keras.utils import np_utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from attacks.fgsm import fgsm


batch_size = 64
nb_classes = 10
nb_epoch = 100

img_channels = 3
img_rows = 32
img_cols = 32

model_saved = True
adv_saved = True


(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


if not model_saved:
    # if we need to train the model, we augment the training data

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zca_whitening=False,
        vertical_flip=False)

    batch = 0
    for X_batch, y_batch in datagen.flow(X_train, y_train,
                                         batch_size=2048):
        print(batch, end=' ', flush=True)
        X_train = np.vstack((X_train, X_batch))
        y_train = np.vstack((y_train, y_batch))
        batch += 1
        if X_train.shape[0] >= 100000:
            break

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')


y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

print('Building model')
sess = tf.InteractiveSession()
K.set_session(sess)


if model_saved:
    print('loading model')
    model = load_model('model/ex_06.h5')
else:
    print('building model')
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=X_train.shape[1:]))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Convolution2D(32, 3, 3))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Convolution2D(64, 3, 3))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Convolution2D(128, 3, 3))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    earlystopping = EarlyStopping(monitor='val_loss', patience=10,
                                  verbose=1)
    model.fit(X_train, y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_split=0.1,
              callbacks=[earlystopping])

    os.makedirs('model', exist_ok=True)
    model.save('model/ex_06.h5')


x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                      img_channels))
y = tf.placeholder(tf.float32, shape=(None, nb_classes))
eps = tf.placeholder(tf.float32, ())
x_adv = fgsm(model, x, y, nb_epoch=4, eps=0.01)


print('Testing...')
score = model.evaluate(X_test, y_test)
print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))


if adv_saved:
    with gzip.open('data/ex_06.pkl.gz', 'rb') as r:
        X_adv, y_adv, y_pred, _ = pickle.load(r)
        X_adv, y_adv = np.array(X_adv), np.array(y_adv)
        y_pred = np.array(y_pred)
else:
    print('generating adversarial data')
    nb_sample = X_test.shape[0]
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

    print('saving adversarial data')
    y_adv = model.predict(X_adv)
    y_pred = model.predict(X_test)
    with gzip.open('data/ex_06.pkl.gz',  'wb') as w:
        pickle.dump([X_adv.tolist(), y_adv.tolist(),
                     y_pred.tolist(), y_test.tolist()], w)


print('Testing against adversarial test data')
score = model.evaluate(X_adv, y_test)
print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))


z0 = np.argmax(y_test, axis=1)
z1 = np.argmax(y_pred, axis=1)
z2 = np.argmax(y_adv, axis=1)
p1 = np.max(y_pred, axis=1)
p2 = np.max(y_adv, axis=1)

X_tmp = np.empty((10, img_rows, img_cols, img_channels))
y_tmp = np.empty((10,))
z_tmp = np.empty((10,), dtype=np.int32)

for i in range(10):
    print('Target {0}'.format(i))
    ind, = np.where(np.all([z0==i, z1==i, z2!=i, p1>0.8, p2>0.8],
                           axis=0))
    cur = np.random.choice(ind.shape[0])
    cur = ind[cur]
    X_tmp[i] = np.squeeze(X_adv[cur])
    y_tmp[i] = p2[cur]
    z_tmp[i] = z2[cur]

fig = plt.figure(figsize=(10, 1.8))
gs = gridspec.GridSpec(1, 10, wspace=0.1, hspace=0.1)

labels = ["airplane","automobile","bird","cat","deer",
          "dog","frog","horse","ship","truck"]

for i in range(10):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(X_tmp[i], interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('{0}\n{1:.2f}'.format(labels[z_tmp[i]], y_tmp[i]),
                  fontsize=8)

gs.tight_layout(fig)
os.makedirs('img', exist_ok=True)
plt.savefig('img/ex_06.png')
