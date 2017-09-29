import tensorflow as tf


def fgsm(model, x, eps=0.01, epochs=1, clip_min=0., clip_max=1.):
    """
    Fast gradient sign method.

    See https://arxiv.org/abs/1412.6572 and https://arxiv.org/abs/1607.02533 for
    details.  This implements the revised version, since the original FGSM has
    label leaking problem (https://arxiv.org/abs/1611.01236).

    :param model: A wrapper that returns the output as well as logits.
    :param x: The input placeholder.
    :param eps: The scale factor for noise.
    :param epochs: The maximum epoch to run.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.

    :return: A tensor, contains adversarial samples for each input.
    """
    x_adv = tf.identity(x)

    ybar = model(x_adv)
    yshape = tf.shape(ybar)
    ydim = yshape[1]

    indices = tf.argmax(ybar, axis=1)
    target = tf.cond(
        tf.equal(ydim, 1),
        lambda: tf.nn.relu(tf.sign(ybar - 0.5)),
        lambda: tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0))

    eps = tf.abs(eps)

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
                             name='fgsm')
    return x_adv
