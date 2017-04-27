import tensorflow as tf


def tgsm(model, x, y=None, eps=0.01, epochs=1, clip_min=0., clip_max=1.):

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

    x_adv, i = tf.while_loop(_cond, _body, (x_adv, 0),
                             back_prop=False, name='tgsm')
    return x_adv
