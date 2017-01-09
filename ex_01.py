import os
import time

import numpy as np
import tensorflow as tf

import keras as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from attacks import jsma


print('Loading mnist')
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

X_train = np.reshape(X_train, (-1, 784))
X_test = np.reshape(X_test, (-1, 784))

y_train = np.reshape(y_train, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))

# one hot encoding
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# split some validation data from training data
validation_split = 0.1
n = X_train.shape[0]
nb_samples = int(n * (1-validation_split))
X_train, X_val = X_train[:nb_samples], X_train[nb_samples:]
y_train, y_val = y_train[:nb_samples], y_train[nb_samples:]

batch_size = 64
nb_sample = X_train.shape[0]
nb_batch = int(nb_sample / batch_size)
nb_epoch = 20

with tf.Session() as sess:
    K.backend.set_session(sess)

    x = tf.placeholder(tf.float32, shape=(None, 784))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    print('Building model')
    model = Sequential()
    model.add(Dense(512, input_dim=784))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    ybar = model(x)
    acc = K.metrics.categorical_accuracy(y, ybar)
    loss = K.metrics.categorical_crossentropy(y, ybar)
    train_step = tf.train.AdamOptimizer().minimize(loss)

    target = tf.placeholder(tf.int32, shape=())
    x_adv = jsma(model, x, target, delta=1.)

    init = tf.global_variables_initializer()
    sess.run(init)

    print('Training')
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

    print('Testing model accuracy against test data')
    tick = time.time()
    accval, lossval = sess.run([acc, loss], feed_dict={
        x: X_test, y: y_test, K.backend.learning_phase(): 0})
    tock = time.time()
    print('Elapsed {0:.2f}s, loss {1:.4f}, acc {2:.4f}'
          .format(tock-tick, np.mean(lossval), np.mean(accval)))

    print('Construct adversarial images from blank images')
    blank = np.zeros((1, 784))
    digits = np.empty((10, 784))
    for i in range(10):
        tick = time.time()
        adv = sess.run(x_adv, feed_dict={
            x: blank, target: i, K.backend.learning_phase(): 0})
        digits[i] = adv.flatten()
        yval = sess.run(ybar, feed_dict={
            x: adv, K.backend.learning_phase(): 0})
        tock = time.time()
        print('Elapsed {0:.2f}s label {1} ({2:.2f})'
              .format(tock-tick, np.argmax(yval), np.max(yval)))

    print('Saving figures')
    os.makedirs('data', exist_ok=True)
    for i, adv in enumerate(digits):
        img = np.reshape(adv, (28, 28))
        plt.imshow(img, cmap='gray', interpolation='none')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('data/{0}.jpg'.format(i))
