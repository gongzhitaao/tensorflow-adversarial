Adversarial Attack with Tensorflow
==================================

I implemented three [adversarial image][3] crafting algorithms with Tensorflow.

- [Fast gradient sign method (FGSM)][1]
- [Jacobian-based saliency map approach (JSMA)][2]
- Difference saliency map approach

## Code ##

The three attacking algorithms can be found in [**attacks.py**](https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/attacks.py).  All return a Tensorflow operation which could be run through `sess.run(...)`.

## Fun Examples ##

- [**ex_01.py**](https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/ex_01.py) creates adversarial digits 0-9 from blank images via JSMA.  Numbers in brackets is the prediction probability.

  <img src="https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/img/0.jpg" height="80">0 (1.00)
  <img src="https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/img/1.jpg" height="80">1 (0.93)
  <img src="https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/img/2.jpg" height="80">2 (1.00)
  <img src="https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/img/3.jpg" height="80">3 (0.97)
  <img src="https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/img/4.jpg" height="80">4 (1.00)

  <img src="https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/img/5.jpg" height="80">5 (1.00)
  <img src="https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/img/6.jpg" height="80">6 (0.96)
  <img src="https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/img/7.jpg" height="80">7 (1.00)
  <img src="https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/img/8.jpg" height="80">8 (0.98)
  <img src="https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/img/9.jpg" height="80">9 (1.00)

## Related Work ##

- [openai/cleverhans](https://github.com/openai/cleverhans)


[1]: https://arxiv.org/abs/1412.6572
[2]: https://arxiv.org/abs/1511.07528
[3]: http://karpathy.github.io/2015/03/30/breaking-convnets/
