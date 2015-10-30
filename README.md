DeepSpark 1.1.1
====================

A *Neural Network implementation* with Scala, [Breeze](https://github.com/scalanlp/breeze) & [Spark](http://spark.apache.org)

Deep Spark follows [GPL v2 license](http://choosealicense.com/licenses/gpl-2.0/).

# Features

## Network

DeepSpark supports following layered neural network implementation:

* *Fully-connected* Neural Network
* *Fully-connected* Rank-3 Tensor Network
* RBF Network
* Word Embedding Network

## Training Methodology

ScalaNetwork supports following training methodologies:

* Stochastic Gradient Descent w/ L1-, L2-regularization, Momentum.
* [AdaGrad](http://www.magicbroom.info/Papers/DuchiHaSi10.pdf)
* [AdaDelta](http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf)

ScalaNetwork supports following environments:

* Multi-Threaded Training Environment, which gets input from Spark RDD

Also you can add negative examples with `Trainer.setNegativeSampler()`.

## Activation Function

ScalaNetwork supports following activation functions:

* Linear
* Sigmoid
* HyperbolicTangent
* Rectifier
* Softplus
* HardSigmoid
* HardTanh
* Softmax
* LeakyReLU

And for RBF Layer,

* Gaussian RBF
* Inverse Quadratic RBF
* HardGaussian RBF (= Hard Inverse Quadratic RBF)

# Usage

Here is some examples for basic usage.

## Download

Currently ScalaNetwork supports Scala version 2.10 ~ 2.11.

* Stable Release is 1.1.1
 
If you are using SBT, add a dependency as described below:

```scala
libraryDependencies += "com.github.nearbydelta" %% "deepspark" % "1.1.1"
```

If you are using Maven, add a dependency as described below:
```xml
<dependency>
  <groupId>com.github.nearbydelta</groupId>
  <artifactId>deepspark_{your.scala.version}</artifactId>
  <version>1.1.1</version>
</dependency>
```

## Examples

* [Simple XOR Network](https://github.com/nearbydelta/deepspark/blob/master/src/test/scala/TestXOR.scala)

* [Simple Word Embedding Simulation](https://github.com/nearbydelta/deepspark/blob/master/src/test/scala/TestEmbedding.scala)