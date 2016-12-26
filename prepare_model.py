import os
import time

import keras as K
import numpy as np
import tensorflow as tf

from utils_mnist import load_mnist
from utils_mnist import model_mnist_mlp as MLP

print('Loading mnist')
(X_train, y_train,
 X_test, y_test,
 X_val, y_val) = load_mnist(validation_split=0.1)

batch_size = 64
nb_samples = X_train.shape[0]
nb_batches = int(nb_samples / batch_size)

with tf.Session() as sess:
    K.backend.set_session(sess)

    x = tf.placeholder(tf.float32, shape=(None, 784))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    print('Building MLP model')
    model = MLP()
    ybar = model(x)

    logits, = ybar.op.inputs
    loss = tf.nn.softmax_cross_entropy_with_logits(logits, y)
    acc = K.metrics.categorical_accuracy(y, ybar)
    train_step = tf.train.AdadeltaOptimizer(
        learning_rate=0.1, rho=0.95, epsilon=1e-08).minimize(loss)

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    sess.run(init)

    print('Training')
    for epoch in range(100):
        print('Epoch {0} '.format(epoch))

        prev = time.time()
        for batch in range(nb_batches):
            end = min(nb_samples, (batch+1)*batch_size)
            start = end - batch_size

            batchloss, _ = sess.run(
                [loss, train_step], feed_dict={
                    x: X_train[start:end],
                    y: y_train[start:end],
                    K.backend.learning_phase(): 1})
        cur = time.time()
        accval = sess.run(acc, feed_dict={
            x: X_val, y: y_val, K.backend.learning_phase(): 0})

        print('{0:.2f}s, loss {1:.4f}, acc {2:.4f}'
              .format(cur-prev, np.mean(batchloss),
                      np.mean(accval)))
        prev = cur

    os.makedirs('model', exist_ok=True)
    savepath = 'model/mnist_mlp'
    saver.save(sess, savepath)
    print('Save saved to {0}'.format(savepath))
