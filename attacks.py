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


def jsma(model, x, target, nb_epoch=None, delta=1., clip_min=0.,
         clip_max=1., eps=1e-3):

    target = tf.to_int64(target)
    if nb_epoch is None:
        nb_epoch = tf.constant(100)

    def _cond(adv_x, epoch):
        label = tf.argmax(model(adv_x), axis=1)
        return tf.logical_and(label[0] != target[0],
                              epoch < nb_epoch)

    def _body(adv_x, epoch):
        ybar = model(adv_x)

        mask = tf.one_hot(target, 10, on_value=True, off_value=False)
        p_target = tf.boolean_mask(ybar, mask)
        dtarget_dx, = tf.gradients(p_target, adv_x)
        p_others = tf.boolean_mask(ybar, tf.logical_not(mask))
        dothers_dx, = tf.gradients(p_others, adv_x)

        cond = tf.cond(delta > tf.constant(0.),
                       lambda: adv_x < clip_max-eps,
                       lambda: adv_x > clip_min+eps)
        scores = dtarget_dx - dothers_dx
        mask = tf.where(cond)
        scores = tf.gather_nd(scores, mask)
        cand = tf.argmax(scores, axis=0)
        pos = tf.gather(mask, cand)
        dx = tf.one_hot([pos[1]], tf.size(adv_x), on_value=delta,
                        off_value=0.)

        adv_x = tf.stop_gradient(adv_x + dx)

        if (clip_min is not None) and (clip_max is not None):
            adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

        epoch += 1

        return adv_x, epoch

    epoch = tf.Variable(0, tf.int32)
    adv_x, epoch = tf.while_loop(_cond, _body, (x, epoch))

    return adv_x
