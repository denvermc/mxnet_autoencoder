# MXNet autoencoder example for creditcard fraud detextion

Autoencoder networks are a type of unsupervised learning where the target is the input. In other words, autoencoders attempt to learn the identity function:

<a href="https://www.codecogs.com/eqnedit.php?latex=h_{W,b}(x)&space;\approx&space;x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_{W,b}(x)&space;\approx&space;x" title="h_{W,b}(x) \approx x" /></a>

This might seem like a really easy task, but autoenoders attempt to make meaningful reconstructions of data by bottlenecking the system. By constraining the size of the hidden units in the network, meaningful structures of data can be discovered in a similar manner to dimensionality-reduction approaches like PCA.

<img src="Autoencoder636.png" alt="Drawing" style="width: 250px;"/>

Because autoencoders are creating low-dimensional representations of mostly 'normal' data, they do not necessarily need to explicitly model fraudulent/anomalous behaviour. They attempt to reconstruct data trained on 'normal' behaviour. Under the assumption that anomalous behaviour is systematically different from 'normal' data, anomalous behaviour will be evident in the reconstruction error.

We're interested in two processes: the *encoding* and *decoding* of data. These can be represented as:

<a href="https://www.codecogs.com/eqnedit.php?latex=\phi&space;:&space;\chi&space;\rightarrow&space;\digamma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi&space;:&space;\chi&space;\rightarrow&space;\digamma" title="\phi : \chi \rightarrow \digamma" /></a>
<a href="https://www.codecogs.com/eqnedit.php?latex=\psi&space;:&space;\digamma&space;\rightarrow&space;\chi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\psi&space;:&space;\digamma&space;\rightarrow&space;\chi" title="\psi : \digamma \rightarrow \chi" /></a>

If $X$ is the input and $Y$ the output, with $n$ elements:

<a href="https://www.codecogs.com/eqnedit.php?latex=Y&space;=&space;(\psi&space;\circ&space;\phi)X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Y&space;=&space;(\psi&space;\circ&space;\phi)X" title="Y = (\psi \circ \phi)X" /></a>

with a mean square error loss function:

<a href="https://www.codecogs.com/eqnedit.php?latex=MSE&space;=&space;\frac{1}{n}\sum\limits_{i=1}^n&space;i^2&space;=&space;(Y_i&space;-&space;\hat{Y}_i)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?MSE&space;=&space;\frac{1}{n}\sum\limits_{i=1}^n&space;i^2&space;=&space;(Y_i&space;-&space;\hat{Y}_i)^2" title="MSE = \frac{1}{n}\sum\limits_{i=1}^n i^2 = (Y_i - \hat{Y}_i)^2" /></a>

By setting a threshold on the reconstruction error (typically $1.96 \sigma$, though thresholds can be set using traditional methods such as ROC curves or precision/recall optimization), anomalous data points can be identified.

Additional information on autoencoders (in MXNet and other frameworks) can be found [here](https://github.com/apache/incubator-mxnet/tree/master/example/autoencoder), [here](https://www.oreilly.com/ideas/anomaly-detection-with-apache-mxnet), and [here](http://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf).
