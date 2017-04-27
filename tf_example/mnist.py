import os
import gzip
from urllib.request import urlopen

import numpy as np


def load_data(fpath='/var/tmp/mnist.npz'):

    if not os.path.exists(fpath):
        print('Downloading mnist dataset')
        _mkdata(fpath)

    db = np.load(fpath)
    X_train, y_train = db['X_train'], db['y_train']
    X_test, y_test = db['X_test'], db['y_test']

    return (X_train, y_train), (X_test, y_test)


def _mkdata(fpath):
    _download('/var/tmp/mnist_X_train.gz',
              'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
    _download('/var/tmp/mnist_y_train.gz',
              'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
    _download('/var/tmp/mnist_X_test.gz',
              'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
    _download('/var/tmp/mnist_y_test.gz',
              'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')

    X_train = _extract_images('/var/tmp/mnist_X_train.gz')
    y_train = _extract_labels('/var/tmp/mnist_y_train.gz')
    y_train = np.expand_dims(y_train, 1)
    X_test = _extract_images('/var/tmp/mnist_X_test.gz')
    y_test = _extract_labels('/var/tmp/mnist_y_test.gz')
    y_test = np.expand_dims(y_test, 1)
    np.savez_compressed(fpath, X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def _extract_images(fpath):
    with gzip.GzipFile(fpath) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number {0} in MNIST image file: {1}'.format(magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows*cols*num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols)
    return data


def _extract_labels(fpath):
    with gzip.GzipFile(fpath) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number {0} in MNIST label file: {1}'.format(magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


def _download(fpath, url):
    if os.path.exists(fpath):
        return
    workdir = os.path.dirname(fpath) or '.'
    os.makedirs(workdir, exist_ok=True)
    with urlopen(url) as ret, open(fpath, 'wb') as w:
        w.write(ret.read())


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_data()
    print('X_train shape: {0}'.format(X_train.shape))
    print('y_train shape: {0}'.format(y_train.shape))
    print('X_test shape: {0}'.format(X_test.shape))
    print('y_test shape: {0}'.format(y_test.shape))
