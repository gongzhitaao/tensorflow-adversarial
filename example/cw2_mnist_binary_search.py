"""
CW attack with binary search for best result.

The CW loss is calculated as loss0+eps*loss1.  A randomly select eps value
will not yeild the best result in general.  The author uses binary search to
find the best one.  And each image does need a slightly different eps value.
Note that this is time consuming if you want to do this over the entire
dataset.

I create this demo mainly to demonstrate the correctness of my implementation
of CW.  Since I do not get very good results with hand-chosen eps value and
batched attacking.
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
batch_size = 1

LEARNING_RATE = 1e-2
BINARY_EPOCHS = 10
EPOCHS = 1000
BOUND = (1e-6, 1)


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

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    env.adv_train_op, env.xadv, env.noise = cw(model, env.x_fixed,
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
    # print('\nPredicting')
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        # print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = env.sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval


def cw_binary_search(env, image, target, epochs=1, bound=(1e-8, 1e2),
                     binary_epochs=10):
    """
    Search for the best result on one image sample.
    """
    print('\nMaking adversarials via CW')

    mindist = float('inf')
    advimg, advprob = None, -1

    lo, hi = bound
    for epoch in range(binary_epochs):
        eps = lo + (hi - lo) / 2

        with Timer('Epoch {0}/{1} lo: {2:g} hi: {3:g}'
                   .format(epoch+1, binary_epochs, lo, hi)):
            feed_dict = {
                env.x_fixed: image,
                env.adv_eps: eps,
                env.adv_y: target}

            # reset the noise before every iteration
            env.sess.run(env.noise.initializer)
            for epoch in range(epochs):
                env.sess.run(env.adv_train_op, feed_dict=feed_dict)

            xadv = env.sess.run(env.xadv, feed_dict=feed_dict)
            ybar = predict(env, xadv, 1)
            label, prob = np.argmax(ybar), np.max(ybar)

            if label != target:
                lo = eps
                if lo > bound[1]:
                    eps *= 10
            else:
                dist = np.linalg.norm(xadv.flatten()-image.flatten())
                if mindist < 0 or dist < mindist:
                    mindist = dist
                    advimg = xadv
                    advprob = prob
                hi = eps

    print('Min distance: {:g}'.format(mindist))
    return advimg, advprob


print('\nTraining')

train(env, X_train, y_train, X_valid, y_valid, load=False, epochs=5,
      name='mnist')

print('\nEvaluating on clean data')

evaluate(env, X_test, y_test)

print('\nGenerating adversarial data')

z0 = np.argmax(y_test, axis=1)
y1 = predict(env, X_test)
z1 = np.argmax(y1, axis=1)

ind = z0 = z1

X_test = X_test[ind]
y_test = y_test[ind]

X_tmp = np.zeros((n_classes, img_size, img_size))
y_tmp = np.zeros(n_classes)

ind = np.random.choice(X_test.shape[0])
image = np.expand_dims(X_test[ind], axis=0)
label = np.argmax(y_test[ind])
y_tmp[label] = np.max(y1[ind])

for i in range(n_classes):
    print('Label {0} --> {1}'.format(label, i))
    if i == label:
        X_tmp[i] = np.squeeze(image)
        continue

    xadv, yadv = cw_binary_search(env, image, target=i, epochs=EPOCHS,
                                  bound=BOUND, binary_epochs=10)
    if yadv > 0:
        X_tmp[i] = np.squeeze(xadv)
    y_tmp[i] = yadv

# Plot it!!

fig = plt.figure(figsize=(n_classes+0.2, 2.2))
gs = gridspec.GridSpec(2, n_classes+1, width_ratios=[1]*n_classes + [0.1],
                       wspace=0.01, hspace=0.01)

for i in range(n_classes):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(X_tmp[i], cmap='gray', interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, i])
    img = ax.imshow(X_tmp[i]-X_tmp[label], cmap='RdBu_r', vmin=-1,
                    vmax=1, interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel('{0} ({1:.2f})'.format(i, y_tmp[i]), fontsize=12)

ax = fig.add_subplot(gs[1, n_classes])
dummy = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=-1,
                                                                vmax=1))
dummy.set_array([])
fig.colorbar(mappable=dummy, cax=ax, ticks=[-1, 0, 1], ticklocation='right')

print('\nSaving figure')

gs.tight_layout(fig)
os.makedirs('img', exist_ok=True)
plt.savefig('img/cw2_mnist_binary_search.png')
