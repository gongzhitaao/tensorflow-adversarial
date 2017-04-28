import os
# supress tensorflow logging other than errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from attacks.smda import smda
import mnist


img_rows = 28
img_cols = 28
img_chas = 1
input_shape = (img_rows, img_cols, img_chas)
n_classes = 10


print('\nLoading mnist')
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

X_train = X_train.reshape(-1, img_rows, img_cols, img_chas)
X_test = X_test.reshape(-1, img_rows, img_cols, img_chas)

# one hot encoding
def _to_categorical(x, n_classes):
    x = np.array(x, dtype=int).ravel()
    n = x.shape[0]
    ret = np.zeros((n, n_classes))
    ret[np.arange(n), x] = 1
    return ret

y_train = _to_categorical(y_train, n_classes)
y_test = _to_categorical(y_test, n_classes)

print('\nShuffling training data')
ind = np.random.permutation(X_train.shape[0])
X_train, y_train = X_train[ind], y_train[ind]

# split training/validation dataset
validation_split = 0.1
n_train = int(X_train.shape[0]*(1-validation_split))
X_train, X_valid = X_train[:n_train], X_train[n_train:]
y_train, y_valid = y_train[:n_train], y_train[n_train:]

# --------------------------------------------------------------------

def model(x, logits=False, training=False):
    conv0 = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3],
                             padding='same', name='conv0',
                             activation=tf.nn.relu)
    pool0 = tf.layers.max_pooling2d(conv0, pool_size=[2, 2],
                                    strides=2, name='pool0')
    conv1 = tf.layers.conv2d(pool0, filters=64,
                             kernel_size=[3, 3], padding='same',
                             name='conv1', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2],
                                    strides=2, name='pool1')
    flat = tf.reshape(pool1, [-1, 7*7*64], name='flatten')
    dense = tf.layers.dense(flat, units=128, activation=tf.nn.relu,
                            name='dense')
    dropout = tf.layers.dropout(dense, rate=0.25, training=training,
                                name='dropout')
    logits_ = tf.layers.dense(dropout, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')
    if logits:
        return y, logits_
    return y


# Collect all tensorflow tensors into one "enviroment" to avoid
# accidental overwriting.
class Dummy:
    pass
env = Dummy()

# We need a scope since the inference graph will be reused later
with tf.variable_scope('model'):
    env.x = tf.placeholder(tf.float32, (None, img_rows, img_cols,
                                        img_chas), name='x')
    env.y = tf.placeholder(tf.float32, (None, n_classes), name='y')
    env.training = tf.placeholder(bool, (), name='mode')

    env.ybar, logits = model(env.x, logits=True,
                             training=env.training)

    z = tf.argmax(env.y, axis=1)
    zbar = tf.argmax(env.ybar, axis=1)
    count = tf.cast(tf.equal(z, zbar), tf.float32)
    env.acc = tf.reduce_mean(count, name='acc')

    xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                   logits=logits)
    env.loss = tf.reduce_mean(xent, name='loss')

env.optim = tf.train.AdamOptimizer().minimize(env.loss)

# Note the reuse=True flag
with tf.variable_scope('model', reuse=True):
    env.target = tf.placeholder(tf.int32, ())
    env.x_adv = smda(model, env.x, env.target, epochs=0.1,
                     min_proba=0.8)

# --------------------------------------------------------------------

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

# --------------------------------------------------------------------

def _evaluate(X_data, y_data, env):
    print('\nEvaluating')
    n_sample = X_data.shape[0]
    batch_size = 128
    n_batch = int(np.ceil(n_sample/batch_size))
    loss, acc = 0, 0
    for ind in range(n_batch):
        print(' batch {0}/{1}'.format(ind+1, n_batch), end='\r')
        start = ind*batch_size
        end = min(n_sample, start+batch_size)
        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end],
                       env.training: False})
        loss += batch_loss*batch_size
        acc += batch_acc*batch_size
    loss /= n_sample
    acc /= n_sample
    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


def _predict(X_data, env):
    print('\nPredicting')
    n_sample = X_data.shape[0]
    batch_size = 128
    n_batch = int(np.ceil(n_sample/batch_size))
    yval = np.empty((X_data.shape[0], n_classes))
    for ind in range(n_batch):
        print(' batch {0}/{1}'.format(ind+1, n_batch), end='\r')
        start = ind*batch_size
        end = min(n_sample, start+batch_size)
        batch_y = sess.run(env.ybar, feed_dict={
            env.x: X_data[start:end], env.training: False})
        yval[start:end] = batch_y
    print()
    return yval

# --------------------------------------------------------------------

print('\nTraining')
n_sample = X_train.shape[0]
batch_size = 128
n_batch = int(np.ceil(n_sample/batch_size))
n_epoch = 5
for epoch in range(n_epoch):
    print('Epoch {0}/{1}'.format(epoch+1, n_epoch))
    for ind in range(n_batch):
        print(' batch {0}/{1}'.format(ind+1, n_batch), end='\r')
        start = ind*batch_size
        end = min(n_sample, start+batch_size)
        sess.run(env.optim, feed_dict={env.x: X_train[start:end],
                                       env.y: y_train[start:end],
                                       env.training: True})
    _evaluate(X_valid, y_valid, env)

print('\nTesting against clean data')
_evaluate(X_test, y_test, env)

# --------------------------------------------------------------------

if False:
    print('\nLoading adversarial')
    db = np.load('data/ex_03.npz')
    X_adv, proba = db['X_adv'], db['proba']
else:
    print('\nGenerating adversarial')
    blank = np.zeros((1, 28, 28, 1))
    X_adv = np.empty((10, 28, 28))
    proba = np.empty((10,))
    for i in np.arange(10):
        print('Target label {0}'.format(i), end='', flush=True)
        X_i_adv = sess.run(env.x_adv, feed_dict={
            env.x: blank, env.target: i})
        y_i_adv = _predict(X_i_adv, env)
        y1 = np.max(y_i_adv)
        z1 = np.argmax(y_i_adv)
        X_adv[i] = np.squeeze(X_i_adv)
        proba[i] = y1
        print(' proba: {0:.2f}'.format(y1))

    os.makedirs('data', exist_ok=True)
    np.savez('data/ex_03.npy', X_adv=X_adv, proba=proba)


print('\nGenerating figure')

fig = plt.figure(figsize=(10, 1.8))
gs = gridspec.GridSpec(1, 10, wspace=0.1, hspace=0.1)

for i in range(10):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(X_adv[i], cmap='gray', interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('{0} ({1:.2f})'.format(i, proba[i]), fontsize=12)

gs.tight_layout(fig)
os.makedirs('img', exist_ok=True)
plt.savefig('img/ex_03.png')
