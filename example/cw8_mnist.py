"""
Use CW method to craft adversarial on MNIST.
"""
import os
from timeit import default_timer

import numpy as np

import matplotlib
matplotlib.use('Agg')           # noqa: E402
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf

from attacks import cw


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


img_size = 28
img_chan = 1
n_classes = 10
batch_size = 64


class Timer(object):
    def __init__(self, msg='Starting.....', timer=default_timer, factor=1,
                 fmt="------- elapsed {:.4f}s --------"):
        self.timer = timer
        self.factor = factor
        self.fmt = fmt
        self.end = None
        self.msg = msg

    def __call__(self):
        """
        Return the current time
        """
        return self.timer()

    def __enter__(self):
        """
        Set the start time
        """
        print(self.msg)
        self.start = self()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Set the end time
        """
        self.end = self()
        print(str(self))

    def __repr__(self):
        return self.fmt.format(self.elapsed)

    @property
    def elapsed(self):
        if self.end is None:
            # if elapsed is called in the context manager scope
            return (self() - self.start) * self.factor
        else:
            # if elapsed is called out of the context manager scope
            return (self.end - self.start) * self.factor


print('\nLoading MNIST')

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.reshape(X_train, [-1, img_size, img_size, img_chan])
X_train = X_train.astype(np.float32) / 255
X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
X_test = X_test.astype(np.float32) / 255

to_categorical = tf.keras.utils.to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('\nSpliting data')

ind = np.random.permutation(X_train.shape[0])
X_train, y_train = X_train[ind], y_train[ind]

VALIDATION_SPLIT = 0.1
n = int(X_train.shape[0] * (1-VALIDATION_SPLIT))
X_valid = X_train[n:]
X_train = X_train[:n]
y_valid = y_train[n:]
y_train = y_train[:n]

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


with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x')
    env.y = tf.placeholder(tf.float32, (None, n_classes), name='y')
    env.training = tf.placeholder_with_default(False, (), name='mode')

    env.ybar, logits = model(env.x, logits=True, training=env.training)

    with tf.variable_scope('acc'):
        count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
        env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

    with tf.variable_scope('loss'):
        xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                       logits=logits)
        env.loss = tf.reduce_mean(xent, name='loss')

    with tf.variable_scope('train_op'):
        optimizer = tf.train.AdamOptimizer()
        vs = tf.global_variables()
        env.train_op = optimizer.minimize(env.loss, var_list=vs)

    env.saver = tf.train.Saver()

    # Note here that the shape has to be fixed during the graph construction
    # since the internal variable depends upon the shape.
    env.x_fixed = tf.placeholder(
        tf.float32, (batch_size, img_size, img_size, img_chan),
        name='x_fixed')
    env.adv_eps = tf.placeholder(tf.float32, (), name='adv_eps')
    env.adv_y = tf.placeholder(tf.int32, (), name='adv_y')

    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    env.adv_train_op, env.xadv, env.noise = cw(model, env.x_fixed, ord_='inf',
                                               y=env.adv_y, eps=env.adv_eps,
                                               optimizer=optimizer)

print('\nInitializing graph')

env.sess = tf.InteractiveSession()
env.sess.run(tf.global_variables_initializer())
env.sess.run(tf.local_variables_initializer())


def evaluate(env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = env.sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


def train(env, X_data, y_data, X_valid=None, y_valid=None, epochs=1,
          load=False, shuffle=True, batch_size=128, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        if not hasattr(env, 'saver'):
            return print('\nError: cannot find saver op')
        print('\nLoading saved model')
        return env.saver.restore(env.sess, 'model/{}'.format(name))

    print('\nTrain model')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
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
            env.sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                                  env.y: y_data[start:end],
                                                  env.training: True})
        if X_valid is not None:
            evaluate(env, X_valid, y_valid)

    if hasattr(env, 'saver'):
        print('\n Saving model')
        os.makedirs('model', exist_ok=True)
        env.saver.save(env.sess, 'model/{}'.format(name))


def predict(env, X_data, batch_size=128):
    """
    Do inference by running env.ybar.
    """
    print('\nPredicting')
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = env.sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval


def make_cw(env, X_data, epochs=1, eps=0.1, batch_size=batch_size):
    """
    Generate adversarial via CW optimization.
    """
    print('\nMaking adversarials via CW')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        with Timer('Batch {0}/{1}   '.format(batch + 1, n_batch)):
            end = min(n_sample, (batch+1) * batch_size)
            start = end - batch_size
            feed_dict = {
                env.x_fixed: X_data[start:end],
                env.adv_eps: eps,
                env.adv_y: np.random.choice(n_classes)}

            env.sess.run(env.noise.initializer)
            for epoch in range(epochs):
                env.sess.run(env.adv_train_op, feed_dict=feed_dict)

            xadv = env.sess.run(env.xadv, feed_dict=feed_dict)
            X_adv[start:end] = xadv

    return X_adv


print('\nTraining')

train(env, X_train, y_train, X_valid, y_valid, load=True, epochs=5,
      name='mnist')

print('\nEvaluating on clean data')

evaluate(env, X_test, y_test)

print('\nGenerating adversarial data')

# It takes a while to run through the full dataset, thus, we demo the result
# through a smaller dataset.  We could actually find the best parameter
# configuration on a smaller dataset, and then apply to the full dataset.
n_sample = 128
ind = np.random.choice(X_test.shape[0], size=n_sample)
X_test = X_test[ind]
y_test = y_test[ind]

X_adv = make_cw(env, X_test, eps=3, epochs=100)

print('\nEvaluating on adversarial data')

evaluate(env, X_adv, y_test)

print('\nRandomly sample adversarial data from each category')

y1 = predict(env, X_test)
y2 = predict(env, X_adv)

z0 = np.argmax(y_test, axis=1)
z1 = np.argmax(y1, axis=1)
z2 = np.argmax(y2, axis=1)

ind = np.logical_and(z0 == z1, z1 != z2)
# print('success: ', np.sum(ind))

ind = z0 == z1

X_test = X_test[ind]
X_adv = X_adv[ind]
z1 = z1[ind]
z2 = z2[ind]
y2 = y2[ind]

ind, = np.where(z1 != z2)
cur = np.random.choice(ind, size=n_classes)
X_org = np.squeeze(X_test[cur])
X_tmp = np.squeeze(X_adv[cur])
y_tmp = y2[cur]

fig = plt.figure(figsize=(n_classes+0.2, 3.2))
gs = gridspec.GridSpec(3, n_classes+1, width_ratios=[1]*n_classes + [0.1],
                       wspace=0.01, hspace=0.01)

label = np.argmax(y_tmp, axis=1)
proba = np.max(y_tmp, axis=1)

for i in range(n_classes):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(X_org[i], cmap='gray', interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, i])
    img = ax.imshow(X_tmp[i]-X_org[i], cmap='RdBu_r', vmin=-1,
                    vmax=1, interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[2, i])
    ax.imshow(X_tmp[i], cmap='gray', interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel('{0} ({1:.2f})'.format(label[i], proba[i]), fontsize=12)

ax = fig.add_subplot(gs[1, n_classes])
dummy = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=-1,
                                                                vmax=1))
dummy.set_array([])
fig.colorbar(mappable=dummy, cax=ax, ticks=[-1, 0, 1], ticklocation='right')

print('\nSaving figure')

gs.tight_layout(fig)
os.makedirs('img/', exist_ok=True)
plt.savefig('img/cw8_mnist.png')
