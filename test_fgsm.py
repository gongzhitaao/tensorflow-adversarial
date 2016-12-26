import numpy as np
import keras as K
import tensorflow as tf

from utils_mnist import load_mnist
from utils_mnist import model_mnist_mlp as MLP

from attacks import fgsm


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

    x_adv = fgsm(x, ybar, eps=0.3)

    # initialize UN-initialized variables only
    init = tf.variables_initializer(set(tf.global_variables())-tmp)
    sess.run(init)

    print('Testing against original test data')
    accval, lossval = sess.run([acc, loss], feed_dict={
        x: X_test, y: y_test, K.backend.learning_phase(): 0})
    print('Original test loss: {0:.4f}  acc: {1:.4f}'
          .format(np.mean(lossval), np.mean(accval)))

    print('Construct adversarial iamges')
    X_adv = sess.run(x_adv, feed_dict={
        x: X_test, K.backend.learning_phase(): 0})

    print('Testing against adversarial test data')
    accval, lossval = sess.run([acc, loss], feed_dict={
        x: X_adv, y: y_test, K.backend.learning_phase(): 0})
    print('Adversarial test loss: {0:.4f}  acc: {1:.4f}'
          .format(np.mean(lossval), np.mean(accval)))
