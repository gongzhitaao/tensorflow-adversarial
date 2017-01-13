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
  choice of `eps`.  The follows are some of the adversarial samples.
  Numbers in brackets are true labels.

    ![0(0)](./img/ex_00_0_9.png?raw=true "0(0)")
    ![2(1)](./img/ex_00_0_9.png?raw=true "2(1)")
    ![3(2)](./img/ex_00_0_9.png?raw=true "3(2)")
    ![8(3)](./img/ex_00_0_9.png?raw=true "8(3)")
    ![9(4)](./img/ex_00_0_9.png?raw=true "9(4)")

    ![6(5)](./img/ex_00_0_9.png?raw=true "6(5)")
    ![5(6)](./img/ex_00_0_9.png?raw=true "5(6)")
    ![2(7)](./img/ex_00_0_9.png?raw=true "2(7)")
    ![9(8)](./img/ex_00_0_9.png?raw=true "9(8)")
    ![4(9)](./img/ex_00_0_9.png?raw=true "4(9)")

- [**ex_01.py**](./ex_01.py) creates adversarial digits 0-9 from blank
  images via JSMA.  Numbers in brackets is the prediction probability.

    ![0 (1.00)](.img/ex_01_0.jpg?raw=true "0 (1.00)")
    ![1 (0.93)](.img/ex_01_1.jpg?raw=true "1 (0.93)")
    ![2 (1.00)](.img/ex_01_2.jpg?raw=true "2 (1.00)")
    ![3 (0.97)](.img/ex_01_3.jpg?raw=true "3 (0.97)")
    ![4 (1.00)](.img/ex_01_4.jpg?raw=true "4 (1.00)")

    ![5 (1.00)](.img/ex_01_5.jpg?raw=true "5 (1.00)")
    ![6 (0.96)](.img/ex_01_6.jpg?raw=true "6 (0.96)")
    ![7 (1.00)](.img/ex_01_7.jpg?raw=true "7 (1.00)")
    ![8 (0.98)](.img/ex_01_8.jpg?raw=true "8 (0.98)")
    ![9 (1.00)](.img/ex_01_9.jpg?raw=true "9 (1.00)")

## Related Work ##

- [openai/cleverhans](https://github.com/openai/cleverhans)
