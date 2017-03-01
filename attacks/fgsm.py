import tensorflow as tf


def fgsm(model, x, y=None, eps=0.01, nb_epoch=1, clip_min=0.,
         clip_max=1.):

    eps = tf.abs(tf.constant(eps))
    nb_epoch = tf.constant(nb_epoch)
    clip_min = tf.constant(clip_min)
    clip_max = tf.constant(clip_max)

    x_adv = tf.identity(x)
    ybar = model(x_adv)
    yshape = tf.shape(ybar)
    nb_sample, ydim = yshape[0], yshape[1]

    if isinstance(y, str):
        if 'min' == y:
            indices = tf.argmin(ybar, axis=1)
            eps = -eps
        elif 'max' == y:
            indices = tf.argmax(ybar, axis=1)
    elif isinstance(y, int):
        indices = tf.fill([nb_sample], y)
    elif isinstance(y, list):
        indices = y

    target = tf.one_hot(indices, ydim)

    def _cond(x_adv, i):
        return tf.less(i, nb_epoch)

    def _body(x_adv, i):
        ybar = model(x_adv)
        logits, = ybar.op.inputs
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=target, logits=logits)
        grad, = tf.gradients(loss, x_adv)
        x_adv = tf.stop_gradient(x_adv + eps*tf.sign(grad))
        return x_adv, i+1

    i = tf.Variable(0)
    x_adv, i = tf.while_loop(_cond, _body, (x_adv, i))
    x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)

    return x_adv
