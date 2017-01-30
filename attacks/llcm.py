import tensorflow as tf


def llcm(model, x, nb_epoch=1, eps=0.01, clip_min=0., clip_max=1.):
    x_adv = tf.identity(x)
    ybar = model(x_adv)
    n = tf.shape(ybar)
    n = n[1]
    yll = tf.argmin(ybar, axis=1)
    yll = tf.one_hot(yll, n)

    def _cond(x_adv, i):
        return tf.less(i, nb_epoch)

    def _body(x_adv, i):
        ybar = model(x_adv)
        logits, = ybar.op.inputs
        loss = tf.nn.softmax_cross_entropy_with_logits(logits, yll)
        grad, = tf.gradients(loss, x_adv)
        x_adv = tf.stop_gradient(x_adv - eps*tf.sign(grad))
        x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)
        i += 1
        return x_adv, i

    i = tf.Variable(0)
    x_adv, i = tf.while_loop(_cond, _body, (x_adv, i))
    return x_adv
