# Highlights <code>[0/2]</code>

-   [ ] [<a href="#fawzi2016-robustness">8</a>]
-   [ ] [<a href="#szegedy2013-intriguing">48</a>]


# Related <code>[17/53]</code>

-   [ ] [<a href="#baluja2017-adversarial">1</a>]
-   [X] [<a href="#bradshaw2017-adversarial">2</a>] DNNs combined with gaussian processes are
    shown to be more robust to adversarial examples.
-   [X] [<a href="#carlini2016-towards">3</a>] proposes another adversarial attacking method.
-   [-] [<a href="#cisse2017-parseval">4</a>] TODO
-   [X] [<a href="#dalvi2004-adversarial">5</a>] assumes that training data may be adversarially
    manipulated by attackers, e.g., spam/fraud/intrusion detection.  They view
    formulate this as a game between the defender and attacker.
-   [X] [<a href="#dong2017-towards">6</a>] leverages adversarial examples to help interpret the
    mechanism of DNN.
-   [ ] [<a href="#fawzi2015-analysis">7</a>]
-   [-] [<a href="#gao2017-deepcloak">9</a>] masks off "redundant" features to defend against
    adversarials.  However it also limits the model's ability to generalize.
-   [-] [<a href="#gondara2017-detecting">10</a>] density ratio of real-real is close to 1, while
    real-adversarial is far away from 1.
-   [X] [<a href="#goodfellow2014-explaining">11</a>] hypothesizes that neural networks are too
    linear to resist linear adversarial perturbation, e.g., FGSM.
-   [-] [<a href="#grosse2017-detection">12</a>]
-   [X] [<a href="#gu2014-towards">13</a>] tried (de-noise) autoencoder to recover adversarial
    samples.  Despite that their experiment looks promising, they only use MNIST
    as benchmark.
-   [X] [<a href="#hayes2017-machine">14</a>] trains end-to-end an attacking model to
    automatically generate adversarial samples in *black-box attack* settings,
    instead of relying on transferability of adversarial samples.
-   [X] [<a href="#he2017-adversarial">15</a>] implies that ensemble of weak defenses is not
    sufficient to provide strong defense against adversarial examples.
-   [X] [<a href="#huang2011-adversarial">16</a>] proposes taxonomy of adversarial machine
    learning.  And the authors formulate it as a game between defender and
    attacker.  It is a high level discussion about adversarial machine learning.
-   [X] [<a href="#huang2015-learning">17</a>] proposes a min-max training procedure to enhance
    the model robustness.  Basically maximize the least perturbation needed to
    generate adversarial samples from each data points.
-   [X] [<a href="#jia2017-adversarial">18</a>] generate adversarial paragraphs in a naive way.
-   [ ] [<a href="#kos2017-adversarial">19</a>]
-   [ ] [<a href="#kos2017-delving">20</a>]
-   [ ] [<a href="#kurakin2016-adversarial">21</a>]
-   [ ] [<a href="#kurakin2016-adversarial-1">22</a>]
-   [ ] [<a href="#liang2017-detecting">23</a>]
-   [ ] [<a href="#lin2017-tactics">24</a>]
-   [ ] [<a href="#liu2017-delving">25</a>]
-   [ ] [<a href="#lowd2005-adversarial">26</a>]
-   [ ] [<a href="#lu2017-safetynet">27</a>]
-   [ ] [<a href="#madry2017-towards">28</a>]
-   [ ] [<a href="#meng2017-magnet">29</a>]
-   [ ] [<a href="#metzen2017-detecting">30</a>]
-   [ ] [<a href="#metzen2017-universal">31</a>]
-   [ ] [<a href="#miyato2015-distributional">32</a>]
-   [ ] [<a href="#moosavi-dezfooli2016-universal">33</a>]
-   [ ] [<a href="#mopuri2017-fast">34</a>]
-   [ ] [<a href="#na2017-cascade">35</a>]
-   [ ] [<a href="#norton2017-adversarial">36</a>]
-   [X] [<a href="#papernot2015-distillation">41</a>] smoothes out the gradient around
    data samples with distilling technique which successfully enhance
    the model's resilience to adversarial noise with minimum impact on
    the model's performance.
-   [X] [<a href="#papernot2015-limitations">38</a>] shows that with a small change in
    pixel intensity, most images in MNIST can be crafted to a desired
    target category different from its actual one.
-   [X] [<a href="#papernot2016-crafting">39</a>] successfully applies fast gradient
    method and forward derivative method to RNN classifiers.
-   [X] [<a href="#papernot2016-practical">40</a>] introduces a *black-box* attack
    against oracle systems, i.e., the attacker has only access to the
    target system's output, by leveraging the *transferability* of
    adversarial samples.  In addition also it demonstrate that the
    attack also applies to non-DNN systems, specifically \(k\)NN,
    however with much less success rate.
-   [X] [<a href="#papernot2016-transferability">37</a>] shows that the
    *transferability* of adversarial samples is not limited to the same
    class of models, but rather extend across different model
    techniques, e.g., deep network, support vector machine, logistic
    regression, decision tree and ensembles.
-   [ ] [<a href="#park2017-adversarial">42</a>]
-   [ ] [<a href="#rozsa2017-adversarial">43</a>]
-   [ ] [<a href="#sabour2015-adversarial">44</a>]
-   [ ] [<a href="#samanta2017-towards">45</a>]
-   [ ] [<a href="#sengupta2017-securing">46</a>]
-   [ ] [<a href="#song2017-multi">47</a>]
-   [X] [<a href="#tabacof2015-exploring">49</a>] shows that adversarial images appear
    in large and dense regions in the pixel space.
-   [ ] [<a href="#tramer2017-ensemble">50</a>]
-   [ ] [<a href="#tramer2017-space">51</a>]
-   [ ] [<a href="#wang2016-theoretical">52</a>]
-   [ ] [<a href="#wang2017-analyzing">53</a>]
-   [ ] [<a href="#xie2017-adversarial">54</a>]
-   [ ] [<a href="#xu2009-robustness">55</a>]


<table>

<tr>
<td>
[<a name="baluja2017-adversarial">1</a>]
</td>
<td>
Shumeet Baluja and Ian Fischer.
 Adversarial transformation networks: Learning to generate adversarial
  examples.
 abs/1703.09387, 2017.
[&nbsp;<a href="http://arxiv.org/abs/1703.09387">http</a>&nbsp;]

</td>
</tr>


<tr>
<td>
[<a name="bradshaw2017-adversarial">2</a>]
</td>
<td>
John Bradshaw, Alexander G de&nbsp;G Matthews, and Zoubin Ghahramani.
 Adversarial examples, uncertainty, and transfer testing robustness in
  gaussian process hybrid deep networks.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="carlini2016-towards">3</a>]
</td>
<td>
Nicholas Carlini and David Wagner.
 Towards evaluating the robustness of neural networks.
 abs/1608.04644, 2016.
[&nbsp;<a href="http://arxiv.org/abs/1608.04644">http</a>&nbsp;]

</td>
</tr>


<tr>
<td>
[<a name="cisse2017-parseval">4</a>]
</td>
<td>
Moustapha Cisse, Piotr Bojanowski, Edouard Grave, Yann Dauphin, and Nicolas
  Usunier.
 Parseval networks: Improving robustness to adversarial examples.
 In <em>International Conference on Machine Learning</em>, pages
  854--863, 2017.

</td>
</tr>


<tr>
<td>
[<a name="dalvi2004-adversarial">5</a>]
</td>
<td>
Nilesh Dalvi, Pedro Domingos, Sumit Sanghai, Deepak Verma, et&nbsp;al.
 Adversarial classification.
 In <em>Proceedings of the tenth ACM SIGKDD international conference
  on Knowledge discovery and data mining</em>, pages 99--108. ACM, 2004.

</td>
</tr>


<tr>
<td>
[<a name="dong2017-towards">6</a>]
</td>
<td>
Yinpeng Dong, Hang Su, Jun Zhu, and Fan Bao.
 Towards interpretable deep neural networks by leveraging adversarial
  examples.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="fawzi2015-analysis">7</a>]
</td>
<td>
Alhussein Fawzi, Omar Fawzi, and Pascal Frossard.
 Analysis of classifiers' robustness to adversarial perturbations.
 abs/1502.02590, 2015.
[&nbsp;<a href="http://arxiv.org/abs/1502.02590">http</a>&nbsp;]

</td>
</tr>


<tr>
<td>
[<a name="fawzi2016-robustness">8</a>]
</td>
<td>
Alhussein Fawzi, Seyed-Mohsen Moosavi-Dezfooli, and Pascal Frossard.
 Robustness of classifiers: from adversarial to random noise.
 In <em>Advances in Neural Information Processing Systems</em>, pages
  1624--1632, 2016.

</td>
</tr>


<tr>
<td>
[<a name="gao2017-deepcloak">9</a>]
</td>
<td>
Ji&nbsp;Gao, Beilun Wang, Zeming Lin, Weilin Xu, and Yanjun Qi.
 Deepcloak: Masking deep neural network models for robustness against
  adversarial samples.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="gondara2017-detecting">10</a>]
</td>
<td>
Lovedeep Gondara.
 Detecting adversarial samples using density ratio estimates.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="goodfellow2014-explaining">11</a>]
</td>
<td>
I.&nbsp;J. Goodfellow, J.&nbsp;Shlens, and C.&nbsp;Szegedy.
 Explaining and Harnessing Adversarial Examples.
 December 2014.
[&nbsp;<a href="http://arxiv.org/abs/1412.6572">arXiv</a>&nbsp;]

</td>
</tr>


<tr>
<td>
[<a name="grosse2017-detection">12</a>]
</td>
<td>
Kathrin Grosse, Praveen Manoharan, Nicolas Papernot, Michael Backes, and
  Patrick McDaniel.
 On the (statistical) detection of adversarial examples.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="gu2014-towards">13</a>]
</td>
<td>
Shixiang Gu and Luca Rigazio.
 Towards deep neural network architectures robust to adversarial
  examples.
 abs/1412.5068, 2014.
[&nbsp;<a href="http://arxiv.org/abs/1412.5068">http</a>&nbsp;]

</td>
</tr>


<tr>
<td>
[<a name="hayes2017-machine">14</a>]
</td>
<td>
Jamie Hayes and George Danezis.
 Machine learning as an adversarial service: Learning black-box
  adversarial examples.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="he2017-adversarial">15</a>]
</td>
<td>
Warren He, James Wei, Xinyun Chen, Nicholas Carlini, and Dawn Song.
 Adversarial example defenses: Ensembles of weak defenses are not
  strong.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="huang2011-adversarial">16</a>]
</td>
<td>
Ling Huang, Anthony&nbsp;D Joseph, Blaine Nelson, Benjamin&nbsp;IP Rubinstein, and
  JD&nbsp;Tygar.
 Adversarial machine learning.
 In <em>Proceedings of the 4th ACM workshop on Security and
  artificial intelligence</em>, pages 43--58. ACM, 2011.

</td>
</tr>


<tr>
<td>
[<a name="huang2015-learning">17</a>]
</td>
<td>
Ruitong Huang, Bing Xu, Dale Schuurmans, and Csaba Szepesv&aacute;ri.
 Learning with a strong adversary.
 abs/1511.03034, 2015.
[&nbsp;<a href="http://arxiv.org/abs/1511.03034">http</a>&nbsp;]

</td>
</tr>


<tr>
<td>
[<a name="jia2017-adversarial">18</a>]
</td>
<td>
Robin Jia and Percy Liang.
 Adversarial examples for evaluating reading comprehension systems.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="kos2017-adversarial">19</a>]
</td>
<td>
J.&nbsp;Kos, I.&nbsp;Fischer, and D.&nbsp;Song.
 Adversarial Examples for Generative models.
 February 2017.
[&nbsp;<a href="http://arxiv.org/abs/1702.06832">arXiv</a>&nbsp;]

</td>
</tr>


<tr>
<td>
[<a name="kos2017-delving">20</a>]
</td>
<td>
J.&nbsp;Kos and D.&nbsp;Song.
 Delving Into Adversarial Attacks on Deep policies.
 May 2017.
[&nbsp;<a href="http://arxiv.org/abs/1705.06452">arXiv</a>&nbsp;]

</td>
</tr>


<tr>
<td>
[<a name="kurakin2016-adversarial">21</a>]
</td>
<td>
A.&nbsp;Kurakin, I.&nbsp;Goodfellow, and S.&nbsp;Bengio.
 Adversarial Examples in the Physical world.
 July 2016.
[&nbsp;<a href="http://arxiv.org/abs/1607.02533">arXiv</a>&nbsp;]

</td>
</tr>


<tr>
<td>
[<a name="kurakin2016-adversarial-1">22</a>]
</td>
<td>
Alexey Kurakin, Ian&nbsp;J. Goodfellow, and Samy Bengio.
 Adversarial machine learning at scale.
 abs/1611.01236, 2016.
[&nbsp;<a href="http://arxiv.org/abs/1611.01236">http</a>&nbsp;]

</td>
</tr>


<tr>
<td>
[<a name="liang2017-detecting">23</a>]
</td>
<td>
Bin Liang, Hongcheng Li, Miaoqiang Su, Xirong Li, Wenchang Shi, and Xiaofeng
  Wang.
 Detecting adversarial examples in deep networks with adaptive noise
  reduction.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="lin2017-tactics">24</a>]
</td>
<td>
Yen-Chen Lin, Zhang-Wei Hong, Yuan-Hong Liao, Meng-Li Shih, Ming-Yu Liu, and
  Min Sun.
 Tactics of adversarial attack on deep reinforcement learning agents.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="liu2017-delving">25</a>]
</td>
<td>
Yanpei Liu, Xinyun Chen, Chang Liu, and Dawn Song.
 Delving into transferable adversarial examples and black-box attacks.
 abs/1611.02770, 2017.
[&nbsp;<a href="http://arxiv.org/abs/1611.02770">http</a>&nbsp;]

</td>
</tr>


<tr>
<td>
[<a name="lowd2005-adversarial">26</a>]
</td>
<td>
Daniel Lowd and Christopher Meek.
 Adversarial learning.
 In <em>Proceedings of the eleventh ACM SIGKDD international
  conference on Knowledge discovery in data mining</em>, pages 641--647. ACM, 2005.

</td>
</tr>


<tr>
<td>
[<a name="lu2017-safetynet">27</a>]
</td>
<td>
Jiajun Lu, Theerasit Issaranon, and David Forsyth.
 Safetynet: Detecting and rejecting adversarial examples robustly.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="madry2017-towards">28</a>]
</td>
<td>
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and
  Adrian Vladu.
 Towards deep learning models resistant to adversarial attacks.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="meng2017-magnet">29</a>]
</td>
<td>
Dongyu Meng and Hao Chen.
 Magnet: a two-pronged defense against adversarial examples.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="metzen2017-detecting">30</a>]
</td>
<td>
Jan&nbsp;Hendrik Metzen, Tim Genewein, Volker Fischer, and Bastian Bischoff.
 On detecting adversarial perturbations.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="metzen2017-universal">31</a>]
</td>
<td>
Jan&nbsp;Hendrik Metzen, Mummadi&nbsp;Chaithanya Kumar, Thomas Brox, and Volker Fischer.
 Universal adversarial perturbations against semantic image
  segmentation.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="miyato2015-distributional">32</a>]
</td>
<td>
Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, Ken Nakae, and Shin Ishii.
 Distributional smoothing with virtual adversarial training.
 1050:25, 2015.

</td>
</tr>


<tr>
<td>
[<a name="moosavi-dezfooli2016-universal">33</a>]
</td>
<td>
Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Omar Fawzi, and Pascal
  Frossard.
 Universal adversarial perturbations.
 2016.

</td>
</tr>


<tr>
<td>
[<a name="mopuri2017-fast">34</a>]
</td>
<td>
Konda&nbsp;Reddy Mopuri, Utsav Garg, and R&nbsp;Venkatesh Babu.
 Fast feature fool: A data independent approach to universal
  adversarial perturbations.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="na2017-cascade">35</a>]
</td>
<td>
Taesik Na, Jong&nbsp;Hwan Ko, and Saibal Mukhopadhyay.
 Cascade adversarial machine learning regularized with a unified
  embedding.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="norton2017-adversarial">36</a>]
</td>
<td>
Andrew&nbsp;P Norton and Yanjun Qi.
 Adversarial-playground: A visualization suite showing how adversarial
  examples fool deep learning.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="papernot2016-transferability">37</a>]
</td>
<td>
N.&nbsp;Papernot, P.&nbsp;McDaniel, and I.&nbsp;Goodfellow.
 Transferability in Machine Learning: From Phenomena To Black-Box
  Attacks Using Adversarial Samples.
 May 2016.
[&nbsp;<a href="http://arxiv.org/abs/1605.07277">arXiv</a>&nbsp;]

</td>
</tr>


<tr>
<td>
[<a name="papernot2015-limitations">38</a>]
</td>
<td>
Nicolas Papernot, Patrick McDaniel, Somesh Jha, Matt Fredrikson, Z.&nbsp;Berkay
  Celik, and Ananthram Swami.
 The limitations of deep learning in adversarial settings.
 abs/1511.07528, 2015.
[&nbsp;<a href="http://arxiv.org/abs/1511.07528">http</a>&nbsp;]

</td>
</tr>


<tr>
<td>
[<a name="papernot2016-crafting">39</a>]
</td>
<td>
Nicolas Papernot, Patrick McDaniel, Ananthram Swami, and Richard&nbsp;E. Harang.
 Crafting adversarial input sequences for recurrent neural networks.
 abs/1604.08275, 2016.
[&nbsp;<a href="http://arxiv.org/abs/1604.08275">http</a>&nbsp;]

</td>
</tr>


<tr>
<td>
[<a name="papernot2016-practical">40</a>]
</td>
<td>
Nicolas Papernot, Patrick&nbsp;Drew McDaniel, Ian&nbsp;J. Goodfellow, Somesh Jha,
  Z.&nbsp;Berkay Celik, and Ananthram Swami.
 Practical black-box attacks against deep learning systems using
  adversarial examples.
 abs/1602.02697, 2016.
[&nbsp;<a href="http://arxiv.org/abs/1602.02697">http</a>&nbsp;]

</td>
</tr>


<tr>
<td>
[<a name="papernot2015-distillation">41</a>]
</td>
<td>
Nicolas Papernot, Patrick&nbsp;Drew McDaniel, Xi&nbsp;Wu, Somesh Jha, and Ananthram
  Swami.
 Distillation as a defense to adversarial perturbations against deep
  neural networks.
 abs/1511.04508, 2015.
[&nbsp;<a href="http://arxiv.org/abs/1511.04508">http</a>&nbsp;]

</td>
</tr>


<tr>
<td>
[<a name="park2017-adversarial">42</a>]
</td>
<td>
Sungrae Park, Jun-Keon Park, Su-Jin Shin, and Il-Chul Moon.
 Adversarial dropout for supervised and semi-supervised learning.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="rozsa2017-adversarial">43</a>]
</td>
<td>
Andras Rozsa, Manuel G&uuml;nther, and Terrance&nbsp;E Boult.
 Adversarial robustness: Softmax versus openmax.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="sabour2015-adversarial">44</a>]
</td>
<td>
Sara Sabour, Yanshuai Cao, Fartash Faghri, and David&nbsp;J Fleet.
 Adversarial manipulation of deep representations.
 2015.

</td>
</tr>


<tr>
<td>
[<a name="samanta2017-towards">45</a>]
</td>
<td>
Suranjana Samanta and Sameep Mehta.
 Towards crafting text adversarial samples.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="sengupta2017-securing">46</a>]
</td>
<td>
Sailik Sengupta, Tathagata Chakraborti, and Subbarao Kambhampati.
 Securing deep neural nets against adversarial attacks with moving
  target defense.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="song2017-multi">47</a>]
</td>
<td>
Chang Song, Hsin-Pai Cheng, Chunpeng Wu, Hai Li, Yiran Chen, and Qing Wu.
 A multi-strength adversarial training method to mitigate adversarial
  attacks.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="szegedy2013-intriguing">48</a>]
</td>
<td>
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan,
  Ian&nbsp;J. Goodfellow, and Rob Fergus.
 Intriguing properties of neural networks.
 abs/1312.6199, 2013.
[&nbsp;<a href="http://arxiv.org/abs/1312.6199">http</a>&nbsp;]

</td>
</tr>


<tr>
<td>
[<a name="tabacof2015-exploring">49</a>]
</td>
<td>
Pedro Tabacof and Eduardo Valle.
 Exploring the space of adversarial images.
 2015.

</td>
</tr>


<tr>
<td>
[<a name="tramer2017-ensemble">50</a>]
</td>
<td>
Florian Tram&egrave;r, Alexey Kurakin, Nicolas Papernot, Dan Boneh, and Patrick
  McDaniel.
 Ensemble adversarial training: Attacks and defenses.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="tramer2017-space">51</a>]
</td>
<td>
Florian Tram&egrave;r, Nicolas Papernot, Ian Goodfellow, Dan Boneh, and Patrick
  McDaniel.
 The space of transferable adversarial examples.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="wang2016-theoretical">52</a>]
</td>
<td>
Beilun Wang, Ji&nbsp;Gao, and Yanjun Qi.
 A theoretical framework for robustness of (deep) classifiers under
  adversarial noise.
 2016.

</td>
</tr>


<tr>
<td>
[<a name="wang2017-analyzing">53</a>]
</td>
<td>
Y.&nbsp;Wang, S.&nbsp;Jha, and K.&nbsp;Chaudhuri.
 Analyzing the Robustness of Nearest Neighbors To Adversarial
  Examples.
 June 2017.
[&nbsp;<a href="http://arxiv.org/abs/1706.03922">arXiv</a>&nbsp;]

</td>
</tr>


<tr>
<td>
[<a name="xie2017-adversarial">54</a>]
</td>
<td>
Cihang Xie, Jianyu Wang, Zhishuai Zhang, Yuyin Zhou, Lingxi Xie, and Alan
  Yuille.
 Adversarial examples for semantic segmentation and object detection.
 2017.

</td>
</tr>


<tr>
<td>
[<a name="xu2009-robustness">55</a>]
</td>
<td>
Huan Xu, Constantine Caramanis, and Shie Mannor.
 Robustness and regularization of support vector machines.
 10(Jul):1485--1510, 2009.

</td>
</tr>
</table>
