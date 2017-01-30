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

- [**ex_00.py**](./ex_00.py) trains a simple CNN on MNIST, achieving
  accuracy ~99%.  Then craft with FGSM adversarial samples from test
  data, of which the CNN accuracy drops to ~30% depending on your
  choice of `eps`.

- [**ex_01.py**](./ex_01.py) creates cross label adversarial images
  via saliency map algorithm (JSMA), left image.  For each row, the
  digit in green frame is the natural one based on which others are
  created.

    <img src="./img/ex_01.png" width="50%">
    <img src="./img/ex_02.png" width="50%">

- [**ex_02.py**](./ex_02.py) creates cross label adversarial images
  via paired saliency map algorithm (JSMA2), right image.

- [**ex_03.py**](./ex_03.py) creates digits from blank images via
  saliency different algorithm (SMDA).

    ![ex_03](./img/ex_03.png?raw=true "digits from scratch")

- [**ex_04.py**](./ex_04.py) creates digits from blank images via
  paired saliency map algorithm (JSMA2).

    ![ex_04](./img/ex_04.png?raw=true "digits from scratch")

## Related Work ##

- [openai/cleverhans](https://github.com/openai/cleverhans)
