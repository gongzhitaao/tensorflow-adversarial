import tensorflow as tf


def smda(model, x, y, epochs=1.0, eps=1.0, clip_min=0.0,
         clip_max=1.0, min_proba=0.0):
    xshape = tf.shape(x)
    n = xshape[0]
    target = tf.cond(tf.equal(0, tf.rank(y)),
                     lambda: tf.zeros([n], dtype=tf.int32)+y,
                     lambda: y)

    if isinstance(epochs, float):
        tmp = tf.to_float(tf.size(x[0])) * epochs
        epochs = tf.to_int32(tf.floor(tmp))

    def _fn(i):
        # `xi` is of the shape (1, ....), the first dimension is the
        # number of samples, 1 in this case.  `yi` is just a scalar,
        # denoting the target class index.
        xi = tf.gather(x, [i])
        yi = tf.gather(target, i)

        # `xadv` is of the shape (1, ...), same as xi.
        xadv = _smda_impl(model, xi, yi, epochs=epochs, eps=eps,
                          clip_min=clip_min, clip_max=clip_max,
                          min_proba=min_proba)
        return xadv[0]

    return tf.map_fn(_fn, tf.range(n), dtype=tf.float32,
                     back_prop=False, name='smda_batch')


def _smda_impl(model, xi, yi, epochs, eps=1.0, clip_min=0.0,
               clip_max=1.0, min_proba=0.0):

    def _cond(x_adv, epoch, pixel_mask):
        ybar = tf.reshape(model(x_adv), [-1])
        proba = ybar[yi]
        label = tf.to_int32(tf.argmax(ybar, axis=0))
        return tf.reduce_all([tf.less(epoch, epochs),
                              tf.reduce_any(pixel_mask),
                              tf.logical_or(tf.not_equal(yi, label),
                                            tf.less(proba, min_proba))],
                             name='_smda_step_cond')

    def _body(x_adv, epoch, pixel_mask):
        ybar = model(x_adv)

        y_target = tf.slice(ybar, [0, yi], [-1, 1])
        dy_dx, = tf.gradients(ybar, x_adv)

        dt_dx, = tf.gradients(y_target, x_adv)
        do_dx = tf.subtract(dy_dx, dt_dx)
        score = dt_dx - do_dx

        ind = tf.where(pixel_mask)
        score = tf.gather_nd(score, ind)

        p = tf.argmax(score, axis=0)
        p = tf.gather(ind, p)
        p = tf.expand_dims(p, axis=0)
        p = tf.to_int32(p)
        dx = tf.scatter_nd(p, [eps], tf.shape(x_adv), name='dx')

        x_adv = tf.stop_gradient(x_adv+dx)
        x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)

        epoch += 1
        pixel_mask = tf.cond(tf.greater(eps, 0),
                             lambda: tf.less(x_adv, clip_max),
                             lambda: tf.greater(x_adv, clip_min))

        return x_adv, epoch, pixel_mask

    epoch = tf.Variable(0, tf.int32)
    x_adv = tf.identity(xi)
    pixel_mask = tf.cond(tf.greater(eps, 0),
                         lambda: tf.less(xi, clip_max),
                         lambda: tf.greater(xi, clip_min))

    x_adv, _, _ = tf.while_loop(_cond, _body,
                                (x_adv, epoch, pixel_mask),
                                back_prop=False, name='smda_step')

    return x_adv
