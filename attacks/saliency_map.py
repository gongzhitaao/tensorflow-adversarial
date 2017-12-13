import tensorflow as tf


__all__ = ['jsma']


def jsma(model, x, y=None, epochs=1, eps=1.0, k=1, clip_min=0.0, clip_max=1.0,
         score_fn=lambda t, o: t * tf.abs(o)):
    """
    Jacobian-based saliency map approach.

    See https://arxiv.org/abs/1511.07528 for details.  During each iteration,
    this method finds the pixel (or two pixels) that has the most influence on
    the result (most salient pixel) and add noise to the pixel.

    :param model: A wrapper that returns the output tensor of the model.
    :param x: The input placeholder a 2D or 4D tensor.
    :param y: The desired class label for each input, either an integer or a
              list of integers.
    :param epochs: Maximum epochs to run.  If it is a floating number in [0,
        1], it is treated as the distortion factor, i.e., gamma in the
        original paper.
    :param eps: The noise added to input per epoch.
    :param k: number of pixels to perturb at a time.  Values other than 1 and
              2 will raise an error.
    :param clip_min: The minimum value in output tensor.
    :param clip_max: The maximum value in output tensor.
    :param score_fn: Function to calculate the saliency score.

    :return: A tensor, contains adversarial samples for each input.
    """
    n = tf.shape(x)[0]

    target = tf.cond(tf.equal(0, tf.rank(y)),
                     lambda: tf.zeros([n], dtype=tf.int32) + y,
                     lambda: y)
    target = tf.stack((tf.range(n), target), axis=1)

    if isinstance(epochs, float):
        tmp = tf.to_float(tf.size(x[0])) * epochs
        epochs = tf.to_int32(tf.floor(tmp))

    if 2 == k:
        _jsma_fn = _jsma2_impl
    else:
        _jsma_fn = _jsma_impl

    return _jsma_fn(model, x, target, epochs=epochs, eps=eps,
                    clip_min=clip_min, clip_max=clip_max, score_fn=score_fn)


def _prod(iterable):
    ret = 1
    for x in iterable:
        ret *= x
    return ret


def _jsma_impl(model, x, yind, epochs, eps, clip_min, clip_max, score_fn):

    def _cond(i, xadv):
        return tf.less(i, epochs)

    def _body(i, xadv):
        ybar = model(xadv)

        dy_dx = tf.gradients(ybar, xadv)[0]

        # gradients of target w.r.t input
        yt = tf.gather_nd(ybar, yind)
        dt_dx = tf.gradients(yt, xadv)[0]

        # gradients of non-targets w.r.t input
        do_dx = dy_dx - dt_dx

        c0 = tf.logical_or(eps < 0, xadv < clip_max)
        c1 = tf.logical_or(eps > 0, xadv > clip_min)
        cond = tf.reduce_all([dt_dx >= 0, do_dx <= 0, c0, c1], axis=0)
        cond = tf.to_float(cond)

        # saliency score for each pixel
        score = cond * score_fn(dt_dx, do_dx)

        shape = score.get_shape().as_list()
        dim = _prod(shape[1:])
        score = tf.reshape(score, [-1, dim])

        # find the pixel with the highest saliency score
        ind = tf.argmax(score, axis=1)
        dx = tf.one_hot(ind, dim, on_value=eps, off_value=0.0)
        dx = tf.reshape(dx, [-1] + shape[1:])

        xadv = tf.stop_gradient(xadv + dx)
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)

        return i+1, xadv

    _, xadv = tf.while_loop(_cond, _body, (0, tf.identity(x)),
                            back_prop=False, name='_jsma_batch')

    return xadv


def _jsma2_impl(model, x, yind, epochs, eps, clip_min, clip_max, score_fn):

    def _cond(k, xadv):
        return tf.less(k, epochs)

    def _body(k, xadv):
        ybar = model(xadv)

        dy_dx = tf.gradients(ybar, xadv)[0]

        # gradients of target w.r.t input
        yt = tf.gather_nd(ybar, yind)
        dt_dx = tf.gradients(yt, xadv)[0]

        # gradients of non-targets w.r.t input
        do_dx = dy_dx - dt_dx

        c0 = tf.logical_or(eps < 0, xadv < clip_max)
        c1 = tf.logical_or(eps > 0, xadv > clip_min)
        cond = tf.reduce_all([dt_dx >= 0, do_dx <= 0, c0, c1], axis=0)
        cond = tf.to_float(cond)

        # saliency score for each pixel
        score = cond * score_fn(dt_dx, do_dx)

        shape = score.get_shape().as_list()
        dim = _prod(shape[1:])
        score = tf.reshape(score, [-1, dim])

        a = tf.expand_dims(score, axis=1)
        b = tf.expand_dims(score, axis=2)
        score2 = tf.reshape(a + b, [-1, dim*dim])
        ij = tf.argmax(score2, axis=1)

        i = tf.to_int32(ij / dim)
        j = tf.to_int32(ij) % dim

        dxi = tf.one_hot(i, dim, on_value=eps, off_value=0.0)
        dxj = tf.one_hot(j, dim, on_value=eps, off_value=0.0)
        dx = tf.reshape(dxi + dxj, [-1] + shape[1:])

        xadv = tf.stop_gradient(xadv + dx)
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)

        return k+1, xadv

    _, xadv = tf.while_loop(_cond, _body, (0, tf.identity(x)),
                            back_prop=False, name='_jsma2_batch')
    return xadv
