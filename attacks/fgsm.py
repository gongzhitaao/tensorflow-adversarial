import tensorflow as tf


def fgsm(x, ybar, eps=0.01, clip_min=0., clip_max=1.):
    n = tf.shape(ybar)
    y = tf.argmax(ybar, 1)
    y = tf.one_hot(y, n[1])
    logits, = ybar.op.inputs
    loss = tf.nn.softmax_cross_entropy_with_logits(logits, y)
    grad, = tf.gradients(loss, x)
    adv_x = tf.stop_gradient(x + eps*tf.sign(grad))
    adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
    return adv_x
