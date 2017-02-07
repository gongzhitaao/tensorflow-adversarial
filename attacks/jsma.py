import tensorflow as tf


def jsma(model, x, target, nb_epoch=None, delta=1., clip_min=0.,
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

        score = -dt_dx * do_dx

        cond1 = tf.cond(delta > tf.constant(0.),
                        lambda: x_adv < clip_max,
                        lambda: x_adv > clip_min)
        cond2 = tf.logical_and(dt_dx > 0, do_dx < 0)
        ind = tf.where(tf.logical_and(cond1, cond2))

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

    epoch = tf.Variable(0, tf.int32)
    x_adv, epoch = tf.while_loop(_cond, _body, (x, epoch))
    return x_adv


def jsma2(model, x, target, nb_epoch=None, delta=1., clip_min=0.,
          clip_max=1.):

    if nb_epoch is None:
        nb_epoch = tf.floor_div(tf.size(x), 20)

    def _cond(x_adv, epoch):
        y = tf.reshape(model(x_adv), [-1])
        return tf.logical_and(tf.less(y[target], 0.9),
                              tf.less(epoch, nb_epoch))

    def _body(x_adv, epoch):
        y = model(x_adv)

        mask = tf.one_hot(target, tf.size(y), on_value=True,
                          off_value=False)
        mask = tf.expand_dims(mask, axis=0)
        yt = tf.boolean_mask(y, mask)
        yo = tf.boolean_mask(y, tf.logical_not(mask))
        dt_dx, = tf.gradients(yt, x_adv)
        do_dx, = tf.gradients(yo, x_adv)

        cond = tf.cond(delta > tf.constant(0.),
                       lambda: x_adv < clip_max,
                       lambda: x_adv > clip_min)
        ind = tf.where(cond)
        n = tf.shape(ind)
        n = n[0]

        ind2 = tf.range(n)
        batch_size = tf.constant(100)

        def _maxpair_batch_cond(i0, j0, v0, start):
            return tf.less(start, n)

        def _maxpair_batch_body(i0, j0, v0, start):
            count = tf.reduce_min([batch_size, n-start])
            ind3 = tf.slice(ind2, [start], [count])

            # Selection C(n, 2), e.g., if n=4, a=[0 0 1 0 1 2], b=[1 2
            # 2 3 3 3], the corresponding element in each array makes
            # a pair, i.e., the pair index are store separately.
            a, b = tf.meshgrid(ind3, ind3)
            c = tf.where(tf.less(a, b))
            a, b = tf.gather_nd(a, c), tf.gather_nd(b, c)

            # ii, jj contains indices to pixels
            ii, jj = tf.gather(ind, a), tf.gather(ind, b)

            ti, oi = tf.gather_nd(dt_dx, ii), tf.gather_nd(do_dx, ii)
            tj, oj = tf.gather_nd(dt_dx, jj), tf.gather_nd(do_dx, jj)

            # the gradient of each pair is the sum of individuals
            t, o = ti+tj, oi+oj

            # increase target probability while decrease others
            c = tf.where(tf.logical_and(t>=0, o<=0))
            t, o = tf.gather_nd(t, c), tf.gather_nd(o, c)
            ii, jj = tf.gather_nd(ii, c), tf.gather_nd(jj, c)

            # saliency score
            score = tf.multiply(t, -o)

            # find the max pair in current batch
            p = tf.argmax(score, axis=0)
            v = tf.reduce_max(score, axis=0)
            i, j = tf.gather(ii, p), tf.gather(jj, p)
            i, j = tf.to_int32(i), tf.to_int32(j)

            i0, j0, v0 = tf.cond(tf.greater(v, v0),
                                 lambda: (i, j, v),
                                 lambda: (i0, j0, v0))
            start += batch_size
            return i0, j0, v0, start

        i = tf.to_int32(tf.gather(ind, 0))
        j = tf.to_int32(tf.gather(ind, 0))
        v = tf.Variable(-1.)
        start = tf.Variable(0)

        # Find max saliency pair in batch.  Naive iteration through
        # the pair takes O(n^2).  Vectorized implementation may
        # speedup the running time significantly, at the expense of
        # O(n^2) space.  So Instead we find the max pair with batch
        # max, during each batch we use vectorized implementation.
        i, j, _, _ = tf.while_loop(_maxpair_batch_cond,
                                   _maxpair_batch_body,
                                   (i, j, v, start))

        dx = tf.scatter_nd([i], [delta], tf.shape(x_adv)) +\
             tf.scatter_nd([j], [delta], tf.shape(x_adv))

        x_adv = tf.stop_gradient(x_adv + dx)

        if (clip_min is not None) and (clip_max is not None):
            x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)

        epoch += 1
        return x_adv, epoch

    epoch = tf.Variable(0, tf.int32)
    x_adv, epoch = tf.while_loop(_cond, _body, (x, epoch))
    return x_adv
