import os

import numpy as np

import matplotlib
matplotlib.use('Agg')           # noqa: E402
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf

from attacks import deepfool
import mnist


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(sess, env, X_data, y_data, X_valid=None, y_valid=None, epochs=1,
          load=False, shuffle=True, batch_size=128, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        print('\nLoading saved model')
        return env.saver.restore(sess, 'model/{}'.format(name))

    print('\nTrain model')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))

        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                              env.y: y_data[start:end],
                                              env.training: True})
        if X_valid is not None:
            evaluate(sess, env, X_valid, y_valid)

    if hasattr(env, 'saver'):
        print('\n Saving model')
        os.makedirs('model', exist_ok=True)
        env.saver.save(sess, 'model/{}'.format(name))


def evaluate(sess, env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


def predict(sess, env, X_data, batch_size=128):
    """
    Do inference by running env.ybar.
    """
    print('\nPredicting')
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        batch_y = sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = batch_y
    print()
    return yval


img_size = 28
img_chan = 1
n_classes = 10


print('\nLoading mnist')
(X_train, y_train), (X_test, y_test) = mnist.load_data(
    xshape=[-1, img_size, img_size, img_chan])

print('\nShuffling training data')
ind = np.random.permutation(X_train.shape[0])
X_train, y_train = X_train[ind], y_train[ind]

# split training/validation dataset
VALIDATION_SPLIT = 0.1
n_train = int(X_train.shape[0] * (1-VALIDATION_SPLIT))
X_valid = X_train[n_train:]
X_train = X_train[:n_train]
y_valid = y_train[n_train:]
y_train = y_train[:n_train]


def model(x, logits=False, training=False):
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3], padding='same',
                             activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)
    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3], padding='same',
                             activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)
    with tf.name_scope('flatten'):
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


print('\nConstructing graph')
env = Dummy()

with tf.variable_scope('model'):
    env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x')
    env.y = tf.placeholder(tf.float32, (None, n_classes), name='y')
    env.training = tf.placeholder_with_default(False, (), name='mode')
    env.ybar, logits = model(env.x, logits=True, training=env.training)

    with tf.name_scope('acc'):
        z = tf.argmax(env.y, axis=1)
        zbar = tf.argmax(env.ybar, axis=1)
        count = tf.cast(tf.equal(z, zbar), tf.float32)
        env.acc = tf.reduce_mean(count, name='acc')

    with tf.name_scope('loss'):
        xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                       logits=logits)
        env.loss = tf.reduce_mean(xent, name='loss')

    with tf.name_scope('train_op'):
        env.train_op = tf.train.AdamOptimizer().minimize(env.loss)

    env.saver = tf.train.Saver()

with tf.variable_scope('model', reuse=True):
    env.adv_prob = tf.placeholder(tf.float32, ())
    env.adv_epochs = tf.placeholder(tf.int32, ())
    env.x_adv, env.noise = deepfool(model, env.x, noise=True, ord_=3.4,
                                    epochs=env.adv_epochs,
                                    min_prob=env.adv_prob)

print('\nInitializing graph')
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

train(sess, env, X_train, y_train, X_valid, y_valid, load=True, epochs=5,
      name='mnist')

evaluate(sess, env, X_train, y_train)

print('\nCrafting adversarial')
n_sample = X_test.shape[0]
batch_size = 128
n_batch = int(np.ceil(n_sample/batch_size))
X_adv = np.empty_like(X_test)
for ind in range(n_batch):
    print(' batch {0}/{1}'.format(ind+1, n_batch), end='\r')
    start = ind*batch_size
    end = min(n_sample, start+batch_size)
    tmp = sess.run(env.x_adv, feed_dict={env.x: X_test[start:end],
                                         env.y: y_test[start:end],
                                         env.adv_epochs: 5,
                                         env.adv_prob: 0.9,
                                         env.training: False})
    X_adv[start:end] = tmp

print('\nSaving adversarial')
os.makedirs('data', exist_ok=True)
np.savez('data/deepfool_mnist_adv.npz', X_test=X_test, X_adv=X_adv,
         y_test=y_test)

print('\nTesting against adversarial data')
evaluate(sess, env, X_adv, y_test)

y1 = predict(sess, env, X_test)
y2 = predict(sess, env, X_adv)

z0 = np.argmax(y_test, axis=1)
z1 = np.argmax(y1, axis=1)
z2 = np.argmax(y2, axis=1)

print('\nPlotting results')
fig = plt.figure(figsize=(10, 2.2))
gs = gridspec.GridSpec(2, 10, wspace=0.1, hspace=0.1)

for i in range(10):
    print('Target {0}'.format(i))
    ind, = np.where(np.all([z0 == i, z1 == i, z2 != i], axis=0))
    ind = np.random.choice(ind)
    xcur = [X_test[ind], X_adv[ind]]
    ycur = y2[ind]
    zcur = z2[ind]

    for j in range(2):
        img = np.squeeze(xcur[j])
        ax = fig.add_subplot(gs[j, i])
        ax.imshow(img, cmap='gray', interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel('{0} ({1:.2f})'.format(zcur, ycur[zcur]), fontsize=12)

print('\nSaving figure')
gs.tight_layout(fig)
os.makedirs('img', exist_ok=True)
plt.savefig('img/deepfool_mnist.png')
