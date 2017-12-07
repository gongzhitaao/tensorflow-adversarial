import os

import numpy as np

import tensorflow as tf
from tensorflow.python.client import timeline

from attacks import jsma


img_size = 28
img_chan = 1
n_classes = 10

print('\nConstruction graph')


def model(x, logits=False, training=False):
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
        z = tf.layers.dropout(z, rate=0.25, training=training)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


class Dummy:
    pass


env = Dummy()


with tf.variable_scope('model'):
    env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x')
    env.y = tf.placeholder(tf.float32, (None, n_classes), name='y')
    env.training = tf.placeholder_with_default(False, (), name='mode')

    env.ybar, logits = model(env.x, logits=True, training=env.training)

with tf.variable_scope('model', reuse=True):
    env.target = tf.placeholder(tf.int32, (), name='target')
    env.adv_epochs = tf.placeholder_with_default(20, shape=(), name='epochs')
    env.adv_eps = tf.placeholder_with_default(0.2, shape=(), name='eps')
    env.x_jsma = jsma(model, env.x, env.target, eps=env.adv_eps, k=1,
                      epochs=env.adv_epochs)

print('\nInitializing graph')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

X_data = np.random.random((1000, 28, 28, 1))
batch_size = 128
n_sample = X_data.shape[0]
n_batch = int((n_sample + batch_size - 1) / batch_size)
X_adv = np.empty_like(X_data)

for batch in range(n_batch):
    print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
    start = batch * batch_size
    end = min(n_sample, start + batch_size)
    feed_dict = {
        env.x: X_data[start:end],
        env.target: np.random.choice(n_classes),
        env.adv_epochs: 1000,
        env.adv_eps: 0.1}
    adv = sess.run(env.x_jsma, feed_dict=feed_dict, options=options,
                   run_metadata=run_metadata)
    X_adv[start:end] = adv
    print()

    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('timeline_{}.json'.format(batch), 'w') as f:
        f.write(chrome_trace)
