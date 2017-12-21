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

    y = tf.stop_gradient(model(x))
    ydim = y.get_shape().as_list()[1]
    if ydim > 1:
        _deepfool_fn = _deepfoolx
    else:
        _deepfool_fn = _deepfool2

    def _fn(xi):
        xi = tf.expand_dims(xi, axis=0)
        noise = _deepfool_fn(model, xi, p=p, eta=eta, epochs=epochs,
                             clip_min=clip_min, clip_max=clip_max,
                             min_prob=min_prob)
        return noise[0]

    z = tf.map_fn(_fn, x, dtype=(tf.float32), back_prop=False,
                  name='deepfool')
    xadv = tf.clip_by_value(x + z*(1+eta), clip_min, clip_max)

    if noise:
        return xadv, z
    return xadv


def _prod(iterable):
    ret = 1
    for x in iterable:
        ret *= x
    return ret


def _deepfool2(model, x, p, epochs, eta, clip_min, clip_max, min_prob):
    y0 = tf.stop_gradient(tf.reshape(model(x), [-1])[0])
    y0 = tf.to_int32(tf.greater(y0, 0.5))

    def _cond(i, z):
        xadv = tf.clip_by_value(x + z*(1+eta), clip_min, clip_max)
        y = tf.stop_gradient(tf.reshape(model(xadv), [-1])[0])
        y = tf.to_int32(tf.greater(y, 0.5))
        return tf.logical_and(tf.less(i, epochs), tf.equal(y0, y))

    def _body(i, z):
        xadv = tf.clip_by_value(x + z*(1+eta), clip_min, clip_max)
        y = tf.reshape(model(xadv), [-1])[0]
        g = tf.gradients(y, xadv)[0]
        dx = - y * g / tf.norm(tf.reshape(g, [-1]))
        return i+1, z+dx

    _, noise = tf.while_loop(_cond, _body, [0, tf.zeros_like(x)],
                             name='_deepfool2_impl', back_prop=False)
    return noise


def _deepfoolx(model, x, p, epochs, eta, clip_min, clip_max, min_prob):
    y0 = tf.stop_gradient(model(x))
    y0 = tf.reshape(y0, [-1])
    k0 = tf.argmax(y0)

    ydim = y0.get_shape().as_list()[0]
    xdim = x.get_shape().as_list()[1:]
    xflat = _prod(xdim)

    def _cond(i, z):
        xadv = tf.clip_by_value(x + z*(1+eta), clip_min, clip_max)
        y = tf.reshape(model(xadv), [-1])
        p = tf.reduce_max(y)
        k = tf.argmax(y)
        return tf.logical_and(tf.less(i, epochs),
                              tf.logical_or(tf.equal(k0, k),
                                            tf.less(p, min_prob)))

    def _body(i, z):
        xadv = tf.clip_by_value(x + z*(1+eta), clip_min, clip_max)
        y = tf.reshape(model(xadv), [-1])

        gs = [tf.reshape(tf.gradients(y[i], xadv)[0], [-1])
              for i in range(ydim)]
        g = tf.stack(gs, axis=0)

        yk, yo = y[k0], tf.concat((y[:k0], y[(k0+1):]), axis=0)
        gk, go = g[k0], tf.concat((g[:k0], g[(k0+1):]), axis=0)

        yo.set_shape(ydim - 1)
        go.set_shape([ydim - 1, xflat])

        a = tf.abs(yo - yk)
        b = go - gk
        c = tf.norm(b, axis=1, ord=p)
        score = a / c
        ind = tf.argmin(score)

        si, bi, ci = score[ind], b[ind], c[ind]
        dx = si * tf.pow(tf.abs(bi), p-1) / tf.pow(ci, p-1) * tf.sign(bi)
        dx = tf.reshape(dx, [-1] + xdim)
        return i+1, z+dx

    _, noise = tf.while_loop(_cond, _body, [0, tf.zeros_like(x)],
                             name='_deepfoolx_impl', back_prop=False)
    return noise
