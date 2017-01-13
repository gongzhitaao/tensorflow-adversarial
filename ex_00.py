import os
import time
import gzip
import pickle

import numpy as np
import tensorflow as tf

import keras as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

validation_split = 0.1
n = X_train.shape[0]
nb_samples = int(n * (1-validation_split))
X_train, X_val = X_train[:nb_samples], X_train[nb_samples:]
y_train, y_val = y_train[:nb_samples], y_train[nb_samples:]

batch_size = 64
nb_epoch = 10

sess = tf.InteractiveSession()
K.backend.set_session(sess)

x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
y = tf.placeholder(tf.float32, shape=(None, 10))

print('Building model')

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

ybar = model(x)
acc = K.metrics.categorical_accuracy(y, ybar)
loss = K.metrics.categorical_crossentropy(y, ybar)
train_step = tf.train.AdamOptimizer().minimize(loss)

eps = tf.placeholder_with_default(0.1, ())
x_adv = fgsm(x, ybar, eps)

saver = tf.train.Saver()

init = tf.global_variables_initializer()
sess.run(init)


print('Training')
nb_sample = X_train.shape[0]
nb_batch = int(nb_sample / batch_size)

for epoch in range(nb_epoch):
    print('Epoch {0}/{1}'.format(epoch+1, nb_epoch))
    tick = time.time()
    for batch in range(nb_batch):
        print(' batch {0}/{1}' .format(batch+1, nb_batch),
              end='\r')
        end = min(nb_sample, (batch+1)*batch_size)
        start = end - batch_size
        sess.run([train_step], feed_dict={
            x: X_train[start:end],
            y: y_train[start:end],
            K.backend.learning_phase(): 1})
    tock = time.time()
    accval, lossval = sess.run([acc, loss], feed_dict={
        x: X_val, y: y_val, K.backend.learning_phase(): 0})
    print('Elapsed {0:.2f}s, loss {1:.4f}, acc {2:.4f}'
          .format(tock-tick, np.mean(lossval), np.mean(accval)))

os.makedirs('model', exist_ok=True)
saver.save(sess, 'model/ex_00')

# saver.restore(sess, 'model/ex_00')

print('Testing...')
nb_sample = X_test.shape[0]
nb_batch = int(np.ceil(nb_sample/batch_size))

print('Testing against original test data')
accval, lossval = 0., 0.
tick = time.time()
for batch in range(nb_batch):
    print(' batch {0}/{1}'.format(batch+1, nb_batch), end='\r')
    start = batch * batch_size
    end = min(nb_sample, start+batch_size)
    batchacc, batchloss = sess.run([acc, loss], feed_dict={
        x: X_test[start:end], y: y_test[start:end],
        K.backend.learning_phase(): 0})
    accval += batchacc
    lossval += batchloss
tock = time.time()
print('Elapsed {0:.2f}s, loss {1:.4f}, acc {2:.4f}'
      .format(tock-tick, lossval/nb_batch, accval/nb_batch))


print('Crafting adversarial test data')
tick = time.time()
X_adv = np.empty(X_test.shape)
for batch in range(nb_batch):
    print(' batch {0}/{1}'.format(batch+1, nb_batch), end='\r')
    start = batch * batch_size
    end = min(nb_sample, start+batch_size)
    X_adv[start:end] = sess.run(x_adv, feed_dict={
        x: X_test[start:end], eps: 0.3,
        K.backend.learning_phase(): 0})
tock = time.time()
print('Elapsed {0:.2f}s'.format(tock-tick))


print('Testing against adversarial test data')
accval, lossval = 0., 0.
tick = time.time()
y_adv = np.empty(y_test.shape)
for batch in range(nb_batch):
    print(' batch {0}/{1}'.format(batch+1, nb_batch), end='\r')
    start = batch * batch_size
    end = min(nb_sample, start+batch_size)
    batchy, batchacc, batchloss = sess.run(
        [ybar, acc, loss], feed_dict={
            x: X_adv[start:end], y: y_test[start:end],
            K.backend.learning_phase(): 0})
    y_adv[start:end] = batchy
    accval += batchacc
    lossval += batchloss
tock = time.time()
print('Elapsed {0:.2f}s, loss {1:.4f}, acc {2:.4f}'
      .format(tock-tick, lossval/nb_batch, accval/nb_batch))


print('Saving adversarial MNIST')
tick = time.time()
os.makedirs('data', exist_ok=True)
with gzip.open('data/ex_00.pkl.gz', 'wb') as w:
    pickle.dump([X_adv.tolist(), y_adv.tolist(),
                 np.argmax(y_test, axis=1).tolist()], w)
tock = time.time()
print('Elapsed {0:.2f}s saved to data/ex_00.pkl.gz'.format(tock-tick))
