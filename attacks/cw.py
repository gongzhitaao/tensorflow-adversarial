import tensorflow as tf


__all__ = ['cw']


def cw(model, x, y=None, eps=1.0, ord_=2, T=2,
       optimizer=tf.train.AdamOptimizer(learning_rate=0.1), alpha=0.9,
       min_prob=0, clip=(0.0, 1.0)):
    """CarliniWagner (CW) attack.

    Only CW-L2 and CW-Linf are implemented since I do not see the point of
    embedding CW-L2 in CW-L1.  See https://arxiv.org/abs/1608.04644 for
    details.

    The idea of CW attack is to minimize a loss that comprises two parts: a)
    the p-norm distance between the original image and the adversarial image,
    and b) a term that encourages the incorrect classification of the
    adversarial images.

    Please note that CW is a optimization process, so it is tricky.  There are
    lots of hyper-parameters to tune in order to get the best result.  The
    binary search process for the best eps values is omitted here.  You could
    do grid search to find the best parameter configuration, if you like.  I
    demonstrate binary search for the best result in an example code.

    :param model: The model wrapper.
    :param x: The input clean sample, usually a placeholder.  NOTE that the
              shape of x MUST be static, i.e., fixed when constructing the
              graph.  This is because there are some variables that depends
              upon this shape.
    :param y: The target label.  Set to be the least-likely label when None.
    :param eps: The scaling factor for the second penalty term.
    :param ord_: The p-norm, 2 or inf.  Actually I only test whether it is 2
        or not 2.
    :param T: The temperature for sigmoid function.  In the original paper,
              the author used (tanh(x)+1)/2 = sigmoid(2x), i.e., t=2.  During
              our experiment, we found that this parameter also affects the
              quality of generated adversarial samples.
    :param optimizer: The optimizer used to minimize the CW loss.  Default to
        be tf.AdamOptimizer with learning rate 0.1. Note the learning rate is
        much larger than normal learning rate.
    :param alpha: Used only in CW-L0.  The decreasing factor for the upper
        bound of noise.
    :param min_prob: The minimum confidence of adversarial examples.
        Generally larger min_prob wil lresult in more noise.
    :param clip: A tuple (clip_min, clip_max), which denotes the range of
        values in x.

    :return: A tuple (train_op, xadv, noise).  Run train_op for some epochs to
             generate the adversarial image, then run xadv to get the final
             adversarial image.  Noise is in the sigmoid-space instead of the
             input space.  It is returned because we need to clear noise
             before each batched attacks.
    """
    xshape = x.get_shape().as_list()
    noise = tf.get_variable('noise', xshape, tf.float32,
                            initializer=tf.initializers.zeros)

    # scale input to (0, 1)
    x_scaled = (x - clip[0]) / (clip[1] - clip[0])

    # change to sigmoid-space, clip to avoid overflow.
    z = tf.clip_by_value(x_scaled, 1e-8, 1-1e-8)
    xinv = tf.log(z / (1 - z)) / T

    # add noise in sigmoid-space and map back to input domain
    xadv = tf.sigmoid(T * (xinv + noise))
    xadv = xadv * (clip[1] - clip[0]) + clip[0]

    ybar, logits = model(xadv, logits=True)
    ydim = ybar.get_shape().as_list()[1]

    if y is not None:
        y = tf.cond(tf.equal(tf.rank(y), 0),
                    lambda: tf.fill([xshape[0]], y),
                    lambda: tf.identity(y))
    else:
        # we set target to the least-likely label
        y = tf.argmin(ybar, axis=1, output_type=tf.int32)

    mask = tf.one_hot(y, ydim, on_value=0.0, off_value=float('inf'))
    yt = tf.reduce_max(logits - mask, axis=1)
    yo = tf.reduce_max(logits, axis=1)

    # encourage to classify to a wrong category
    loss0 = tf.nn.relu(yo - yt + min_prob)

    axis = list(range(1, len(xshape)))
    ord_ = float(ord_)

    # make sure the adversarial images are visually close
    if 2 == ord_:
        # CW-L2 Original paper uses the reduce_sum version.  These two
        # implementation does not differ much.

        # loss1 = tf.reduce_sum(tf.square(xadv-x), axis=axis)
        loss1 = tf.reduce_mean(tf.square(xadv-x))
    else:
        # CW-Linf
        tau0 = tf.fill([xshape[0]] + [1]*len(axis), clip[1])
        tau = tf.get_variable('cw8-noise-upperbound', dtype=tf.float32,
                              initializer=tau0, trainable=False)
        diff = xadv - x - tau

        # if all values are smaller than the upper bound value tau, we reduce
        # this value via tau*0.9 to make sure L-inf does not get stuck.
        tau = alpha * tf.to_float(tf.reduce_all(diff < 0, axis=axis))
        loss1 = tf.nn.relu(tf.reduce_sum(diff, axis=axis))

    loss = eps*loss0 + loss1
    train_op = optimizer.minimize(loss, var_list=[noise])

    # We may need to update tau after each iteration.  Refer to the CW-Linf
    # section in the original paper.
    if 2 != ord_:
        train_op = tf.group(train_op, tau)

    return train_op, xadv, noise
