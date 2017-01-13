import tensorflow as tf


def fgsm(x, ybar, eps, clip_min=0., clip_max=1.):
    y = tf.equal(ybar, tf.reduce_max(ybar, 1, keep_dims=True))
    y = tf.to_float(y)
    y = y / tf.reduce_sum(y, 1, keep_dims=True)
    logits, = ybar.op.inputs
    loss = tf.nn.softmax_cross_entropy_with_logits(logits, y)
    grad, = tf.gradients(loss, x)
    signed_grad = tf.sign(grad)
    scaled_signed_grad = eps * signed_grad
    adv_x = tf.stop_gradient(x + scaled_signed_grad)
    adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
    return adv_x
