import tensorflow as tf


def tgsm(model, x, y=None, eps=0.01, epochs=1, clip_min=0., clip_max=1.):
    """
    Target class gradient sign method.

    See https://arxiv.org/pdf/1607.02533.pdf.  This method is similar to FGSM.
    The only difference is that

        1. TGSM allows to specify the desired label, i.e., targeted attack.

        2. Modified towards the least-likely class label when desired label is
           not specified.

    :param model: A model that returns the output as well as logits.
    :param x: The input placeholder.
    :param y: The desired target label, set to the least-likely class if not
              specified.
    :param eps: The noise scale factor.
    :param epochs: Maximum epoch to run.
    :param clip_min: Minimum value in output.
    :param clip_max: Maximum value in output.
    """
    x_adv = tf.identity(x)
    eps = -tf.abs(eps)
    ybar = model(x_adv)
    yshape = tf.shape(ybar)
    ydim = yshape[1]

    if y is None:
        indices = tf.argmin(ybar, axis=1)
    else:
        xshape = tf.shape(x)
        n = xshape[0]
        indices = tf.cond(tf.equal(0, tf.rank(y)),
                          lambda: tf.zeros([n], dtype=tf.int32)+y,
                          lambda: y)

    target = tf.one_hot(indices, ydim, on_value=clip_max,
                        off_value=clip_min)

    def _cond(x_adv, i):
        return tf.less(i, epochs)

    def _body(x_adv, i):
        ybar, logits = model(x_adv, logits=True)
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=target, logits=logits)
        dy_dx, = tf.gradients(loss, x_adv)
        x_adv = tf.stop_gradient(x_adv + eps*tf.sign(dy_dx))
        x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)
        return x_adv, i+1

    x_adv, _ = tf.while_loop(_cond, _body, (x_adv, 0), back_prop=False,
                             name='tgsm')
    return x_adv
