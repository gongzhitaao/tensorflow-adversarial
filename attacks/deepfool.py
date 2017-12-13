import tensorflow as tf


__all__ = ['deepfool']


def deepfool(model, x, noise=False, eta=0.01, ord_=2, epochs=3, clip_min=0.0,
             clip_max=1.0, min_prob=0.0):
    """DeepFool implementation in Tensorflow.

    See https://arxiv.org/abs/1511.04599 for details.

    :param model: Model function.
    :param x: 2D or 4D input tensor.
    :param noise: Also return the noise if True.
    :param eta: Small overshoot value to cross the boundary.
    :param ord_: Which norm to use in computation.
    :param epochs: Maximum epochs to run.
    :param clip_min: Min clip value for output.
    :param clip_max: Max clip value for output.
    :param min_prob: Minimum probability for adversarial samples.

    :return: Adversarials, of the same shape as x.
    """
    if float('Inf') == ord_:
        p = 1.0
    else:
        p = ord_ / (ord_ - 1.0)

    def _fn(xi):
        xi = tf.expand_dims(xi, axis=0)
        xadv, noise = _deepfool_impl(model, xi, p=p, eta=eta, epochs=epochs,
                                     clip_min=clip_min, clip_max=clip_max,
                                     min_prob=min_prob)
        return xadv[0], noise[0]

    xadv, z = tf.map_fn(_fn, x, dtype=(tf.float32, tf.float32),
                        back_prop=False, name='deepfool')

    if noise:
        return xadv, z
    return xadv


def _deepfool_impl(model, x, p, epochs, eta, clip_min, clip_max, min_prob):
    y0 = tf.reshape(model(x), [-1])
    k0 = tf.argmax(y0)

    def _cond(i, x, z):
        y = tf.reshape(model(x), [-1])
        p = tf.reduce_max(y)
        k = tf.argmax(y)
        return tf.logical_and(tf.less(i, epochs),
                              tf.logical_or(tf.equal(k0, k),
                                            tf.less(p, min_prob)))

    def _body(i, x, z):
        y = tf.reshape(model(x), [-1])

        ys = tf.unstack(y)
        gs = [tf.reshape(tf.gradients(yi, x)[0], [-1]) for yi in ys]
        g = tf.stack(gs, axis=0)

        # The following implementation assumes that 1) 0/0 = tf.nan, 2) nan is
        # ignored by tf.argmin().
        a = tf.abs(y - y[k0])
        b = tf.norm(g - g[k0], axis=1, ord=2)
        score = a / b
        ind = tf.argmin(score)

        ai, bi, gi = a[ind], b[ind], g[ind]
        dx = ai / tf.pow(bi, p) * tf.pow(bi, p-1) * gi
        dx = tf.reshape(dx, x.get_shape().as_list())

        x = tf.stop_gradient(x + dx*(1+eta))
        x = tf.clip_by_value(x, clip_min, clip_max)
        z = tf.stop_gradient(z + dx)
        return i+1, x, z

    _, xadv, noise = tf.while_loop(_cond, _body,
                                   [0, tf.identity(x), tf.zeros_like(x)],
                                   name='_deepfool_impl', back_prop=False)
    return xadv, noise
