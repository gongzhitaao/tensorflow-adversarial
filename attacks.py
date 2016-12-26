import tensorflow as tf


def fgsm(x, ybar, eps=0.1, clip_min=0., clip_max=1.):
    y = tf.equal(ybar, tf.reduce_max(ybar, 1, keep_dims=True))
    y = tf.to_float(y)
    y = y / tf.reduce_sum(y, 1, keep_dims=True)
    logits, = ybar.op.inputs
    loss = tf.nn.softmax_cross_entropy_with_logits(logits, y)
    grad, = tf.gradients(loss, x)
    signed_grad = tf.sign(grad)
    scaled_signed_grad = eps * signed_grad
    adv_x = tf.stop_gradient(x + scaled_signed_grad)

    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x


def jsma(x, ybar, target, delta=1., clip_min=0., clip_max=1.,
         eps=1e-3):

    mask = tf.one_hot([target], 10, on_value=True, off_value=False)
    p_target = tf.boolean_mask(ybar, mask)
    dtarget_dx, = tf.gradients(p_target, x)
    p_others = tf.boolean_mask(ybar, tf.logical_not(mask))
    dothers_dx, = tf.gradients(p_others, x)

    cond = tf.cond(delta > tf.constant(0.),
                   lambda: x < clip_max-eps,
                   lambda: x > clip_min+eps)
    scores = dothers_dx - dtarget_dx
    mask = tf.where(cond)
    scores = tf.gather_nd(scores, mask)
    cand = tf.argmin(scores, axis=0)
    pos = tf.gather(mask, cand)
    dx = tf.one_hot([pos[1]], tf.size(x), on_value=delta,
                    off_value=0.)
    adv_x = tf.stop_gradient(x + dx)

    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x
