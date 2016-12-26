Adversarial Attack with Tensorflow
==================================

I implemented two [adversarial image][3] crafting algorithms with Tensorflow.

## Introduction ##

Two algorithms:
- [Fast gradient sign method (FGSM)][1]
- Improved [Jacobian-based saliency map approach (JSMA)][2]

## Code ##

The two algorithms are implemented in [attacks.py](https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/attacks.py).  Both return a Tensorflow operator which could be run through `sess.run(...)`.  Please refer to [test_fgsm.py](https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/test_fgsm.py) and [test_jsma.py](https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/test_jsma.py) for the usage.

## Fun Examples ##

The following adversarial images are generated via JSMA from blank image on a 3-layer MLP trained on MNIST (achieving accuracy 95%).  The labels are all predicted with condfidence over 99% by the network.

<img src="https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/img/0.jpg" height="100">0
<img src="https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/img/1.jpg" height="100">1
<img src="https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/img/2.jpg" height="100">2
<img src="https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/img/3.jpg" height="100">3
<img src="https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/img/4.jpg" height="100">4

<img src="https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/img/5.jpg" height="100">5
<img src="https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/img/6.jpg" height="100">6
<img src="https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/img/7.jpg" height="100">7
<img src="https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/img/8.jpg" height="100">8
<img src="https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/img/9.jpg" height="100">9

## Related Work ##

- [openai/cleverhans](https://github.com/openai/cleverhans)


[1]: https://arxiv.org/abs/1412.6572
[2]: https://arxiv.org/abs/1511.07528
[3]: http://karpathy.github.io/2015/03/30/breaking-convnets/
