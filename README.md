# MXNet autoencoder example for creditcard fraud detextion

Autoencoder networks are a type of unsupervised learning where the target is the input. In other words, autoencoders attempt to learn the identity function:

\begin{equation*}
h_{W,b}(x) \approx x
\end{equation*}

This might seem like a really easy task, but autoenoders attempt to make meaningful reconstructions of data by bottlenecking the system. By constraining the size of the hidden units in the network, meaningful structures of data can be discovered in a similar manner to dimensionality-reduction approaches like PCA.

<img src="Autoencoder636.png" alt="Drawing" style="width: 250px;"/>

Because autoencoders are creating low-dimensional representations of mostly 'normal' data, they do not necessarily need to explicitly model fraudulent/anomalous behaviour. They attempt to reconstruct data trained on 'normal' behaviour. Under the assumption that anomalous behaviour is systematically different from 'normal' data, anomalous behaviour will be evident in the reconstruction error.

We're interested in two processes: the *encoding* and *decoding* of data. These can be represented as:

\begin{equation*}
\phi : \chi \rightarrow \digamma
\end{equation*}
\begin{equation*}
\psi : \digamma \rightarrow \chi
\end{equation*}

If $X$ is the input and $Y$ the output, with $n$ elements:

\begin{equation*}
Y = (\psi \circ \phi)X
\end{equation*}

with a mean square error loss function:

\begin{equation*}
MSE = \frac{1}{n}\sum\limits_{i=1}^n i^2 = (Y_i - \hat{Y}_i)^2
\end{equation*}

By setting a threshold on the reconstruction error (typically $1.96 \sigma$, though thresholds can be set using traditional methods such as ROC curves or precision/recall optimization), anomalous data points can be identified.

Additional information on autoencoders (in MXNet and other frameworks) can be found [here](https://github.com/apache/incubator-mxnet/tree/master/example/autoencoder), [here](https://www.oreilly.com/ideas/anomaly-detection-with-apache-mxnet), and [here](http://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf).
