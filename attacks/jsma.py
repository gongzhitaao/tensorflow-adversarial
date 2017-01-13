import tensorflow as tf


def jsma(model, x, target, nb_epoch=None, delta=1., clip_min=0.,
         clip_max=1.):

    if nb_epoch is None:
        nb_epoch = tf.floor_div(tf.size(x), 20)

    def _cond(adv_x, epoch):
        ybar = tf.reshape(model(adv_x), [-1])
        return tf.logical_and(tf.less(ybar[target], 0.9),
                              tf.less(epoch, nb_epoch))

    def _body(adv_x, epoch):
        y = model(adv_x)

        nb_input = tf.size(adv_x)
        nb_output = tf.size(y)

        mask = tf.one_hot(target, nb_output, on_value=True,
                          off_value=False)
        mask = tf.expand_dims(mask, axis=0)
        yt = tf.boolean_mask(y, mask)
        yo = tf.boolean_mask(y, tf.logical_not(mask))
        dt_dx, = tf.gradients(yt, adv_x)
        do_dx, = tf.gradients(yo, adv_x)

        ind = tf.range(nb_input)
        mask = tf.cond(tf.less(delta, 0.),
                       lambda: tf.greater(adv_x, clip_min),
                       lambda: tf.less(adv_x, clip_max))
        mask = tf.reshape(mask, [-1])
        ind = tf.boolean_mask(ind, mask)
        a, b = tf.meshgrid(ind, ind)
        ind = tf.stack([a, b], axis=2)
        n = tf.shape(ind)
        mask = tf.ones([n[0], n[0]], tf.int32)
        mask = tf.equal(tf.matrix_band_part(mask, -1, 0), 0)
        ind = tf.boolean_mask(ind, mask)

        a, b = tf.meshgrid(dt_dx, dt_dx)
        dt_dx = tf.gather_nd(a+b, ind)

        a, b = tf.meshgrid(do_dx, do_dx)
        do_dx = tf.gather_nd(a+b, ind)

        c = tf.logical_and(tf.greater_equal(dt_dx, 0),
                           tf.less_equal(do_dx, 0))
        ind = tf.boolean_mask(ind, c)
        dt_dx = tf.boolean_mask(dt_dx, c)
        do_dx = tf.boolean_mask(do_dx, c)
        score = -1 * dt_dx * do_dx

        pos = tf.argmax(score, axis=0)
        pos = ind[tf.to_int32(pos)]
        i, j = pos[0], pos[1]

        dx = tf.one_hot(i, nb_input, on_value=delta, off_value=0.) +\
             tf.one_hot(j, nb_input, on_value=delta, off_value=0.)
        dx = tf.expand_dims(dx, axis=0)

        adv_x = tf.stop_gradient(adv_x + dx)

        if (clip_min is not None) and (clip_max is not None):
            adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

        epoch += 1

        return adv_x, epoch

    epoch = tf.Variable(0, tf.int32)
    adv_x, epoch = tf.while_loop(_cond, _body, (x, epoch))
    return adv_x
