import numpy as np
import keras as K
import tensorflow as tf

from utils_mnist import load_mnist
from utils_mnist import model_mnist_mlp as MLP

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from attacks import jsma


print('Loading MNIST')
X_train, y_train, X_test, y_test = load_mnist()

with tf.Session() as sess:
    K.backend.set_session(sess)

    x = tf.placeholder(tf.float32, shape=(None, 784))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    print('Building model')
    model = MLP()
    ybar = model(x)

    tmp = set(tf.global_variables())

    acc = K.metrics.categorical_accuracy(y, ybar)
    loss = K.metrics.categorical_crossentropy(y, ybar)

    saver = tf.train.Saver()
    savepath = 'model/mnist_mlp'
    saver.restore(sess, savepath)

    target = tf.placeholder(tf.int32, shape=(1,))
    x_adv = jsma(model, x, target, delta=0.3)

    # initialize UN-initialized variables only
    init = tf.variables_initializer(set(tf.global_variables())-tmp)
    sess.run(init)

    print('Testing against original test data')
    accval, lossval = sess.run([acc, loss], feed_dict={
        x: X_test, y: y_test, K.backend.learning_phase(): 0})
    print('Original test loss: {0:.4f}  acc: {1:.4f}'
          .format(np.mean(lossval), np.mean(accval)))

    print('Construct adversarial images')

    # Randomly select an image from each category, and modify to any
    # other classes.
    labels = np.argmax(y_test, axis=1)
    for i in range(10):
        indices, = np.where(i == labels)

        while True:
            idx = np.random.choice(indices)
            cand = X_test[idx, np.newaxis]
            yval = sess.run(ybar, feed_dict={
                x: cand, K.backend.learning_phase(): 0})
            if np.argmax(yval) == i:
                break

        for j in range(10):
            if j == i:
                continue

            adv = sess.run(x_adv, feed_dict={
                x: cand, target: [j], K.backend.learning_phase(): 0})
            yval = sess.run(ybar, feed_dict={
                x: adv, K.backend.learning_phase(): 0})
            yadv = np.argmax(yval)
            print('Predicted label for {0}: {1} ({2:.2f})'
                  .format(i, yadv, np.max(yval)))

            plt.imshow(adv.reshape((28, 28)), cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('org{0}-adv{1}.jpg'.format(i, yadv))
