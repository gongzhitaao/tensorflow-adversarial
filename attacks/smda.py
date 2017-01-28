import tensorflow as tf


def smda(model, x, target, nb_epoch=None, delta=1., clip_min=0.,
         clip_max=1.):

    if nb_epoch is None:
        nb_epoch = tf.floor_div(tf.size(x), 20)

    def _cond(x_adv, epoch):
        ybar = tf.reshape(model(x_adv), [-1])
        return tf.logical_and(tf.less(ybar[target], 0.9),
                              tf.less(epoch, nb_epoch))

    def _body(x_adv, epoch):
        y = model(x_adv)

        nb_input = tf.size(x_adv)
        nb_output = tf.size(y)

        mask = tf.one_hot(target, nb_output, on_value=True,
                          off_value=False)
        mask = tf.expand_dims(mask, axis=0)
        yt = tf.boolean_mask(y, mask)
        yo = tf.boolean_mask(y, tf.logical_not(mask))
        dt_dx, = tf.gradients(yt, x_adv)
        do_dx, = tf.gradients(yo, x_adv)

        score = dt_dx - do_dx

        ind = tf.where(tf.cond(delta > tf.constant(0.),
                               lambda: x_adv < clip_max,
                               lambda: x_adv > clip_min))
        score = tf.gather_nd(score, ind)

        p = tf.argmax(score, axis=0)
        p = tf.gather(ind, p)
        p = tf.expand_dims(p, axis=0)
        p = tf.to_int32(p)
        dx = tf.scatter_nd(p, [delta], tf.shape(x_adv))

        x_adv = tf.stop_gradient(x_adv + dx)

        if (clip_min is not None) and (clip_max is not None):
            x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)

        epoch += 1

        return x_adv, epoch

    x_adv = tf.identity(x)
    epoch = tf.Variable(0, tf.int32)
    x_adv, epoch = tf.while_loop(_cond, _body, (x_adv, epoch))
    return x_adv
