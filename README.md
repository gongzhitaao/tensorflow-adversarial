Adversarial Attack with Tensorflow
==================================

I implemented
four
[adversarial image](http://karpathy.github.io/2015/03/30/breaking-convnets/) crafting
algorithms with Tensorflow.

- [Fast Gradient Sign Method (FGSM)](https://arxiv.org/abs/1412.6572)
- [Jacobian-based Saliency Map Approach (JSMA)](https://arxiv.org/abs/1511.07528)
- [Least-Likely Class Method (LLCM)](https://arxiv.org/abs/1607.02533)
- Saliency map difference approach (SMDA)

## Code ##

The four attacking algorithms can be found in [**attacks**](./attacks)
folder.  All return a Tensorflow operation which could be run through
`sess.run(...)`.

## Fun Examples ##

- [**ex_00.py**](./ex_00.py) trains a simple CNN on MNIST, achieving
  accuracy ~99%.  Then craft with FGSM adversarial samples from test
  data, of which the CNN accuracy drops to 0% depending on your choice
  of `eps` and `nb_epoch`.  The original label for the following
  digits are 0 through 9 originally, and the predicted label with
  probability are shown below each digit.

    ![ex_00](./img/ex_00.png?raw=true "fgsm digits")

- [**ex_01.py**](./ex_01.py) creates cross label adversarial images
  via saliency map algorithm (JSMA), left image.  For each row, the
  digit in green frame is the natural one based on which others are
  created.

    <img src="./img/ex_01.png" width="45%">
    <img src="./img/ex_02.png" width="45%">

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
