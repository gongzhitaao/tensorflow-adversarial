Adversarial Attack with Tensorflow
==================================

I implemented
three
[adversarial image](http://karpathy.github.io/2015/03/30/breaking-convnets/) crafting
algorithms with Tensorflow.

- [Fast gradient sign method (FGSM)](https://arxiv.org/abs/1412.6572)
- [Jacobian-based saliency map approach (JSMA)](https://arxiv.org/abs/1511.07528)
- Difference saliency map approach

## Code ##

The three attacking algorithms can be found
in
[**attacks.py**](https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/attacks.py).
All return a Tensorflow operation which could be run through
`sess.run(...)`.

## Fun Examples ##

- [**ex_01.py**](./ex_00.py) trains a simple CNN on MNIST, achieving
  accuracy ~99%.  Then craft with FGSM adversarial samples from test
  data, of which the CNN accuracy drops to ~30% depending on your
  choice of `eps`.

- [**ex_01.py**](./ex_01.py) creates cross label adversarial images.
  For each row, the digit in green frame is the natural one based on
  which others are created.

    ![ex_01](./img/ex_01.png?raw=true "cross label adversarial")

- [**ex_02.py**](./ex_02.py) creates digits from blank images.

    ![ex_02](./img/ex_02.png?raw=true "digits from scratch")

## Related Work ##

- [openai/cleverhans](https://github.com/openai/cleverhans)
