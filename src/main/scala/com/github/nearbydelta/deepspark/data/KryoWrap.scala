package com.github.nearbydelta.deepspark.data

import breeze.linalg.{DenseMatrix, DenseVector}
import com.esotericsoftware.kryo.Serializer
import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.nearbydelta.deepspark.layer._
import com.github.nearbydelta.deepspark.network.{GeneralNetwork, SimpleNetwork}
import com.github.nearbydelta.deepspark.word.{LedgerSGD, LedgerAdaGrad, LedgerAdaDelta, LedgerModel}
import com.github.nearbydelta.deepspark.word.layer._
import com.twitter.chill.{Kryo, ScalaKryoInstantiator}

/**
 * Created by bydelta on 15. 10. 19.
 */
object KryoWrap {
  val kryo = new ScalaKryoInstantiator().newKryo()
  init(kryo)

  def init(kryo: Kryo): Unit = {
    // Please, preserve this ordering.
    kryo.register(classOf[LedgerModel])

    // To register specialized types in scala, instantiate things
    kryo.register(DenseVector(0.0).getClass, DataVectorSerializer)
    kryo.register(DenseMatrix(0.0).getClass, MatrixSerializer)
    kryo.register(classOf[DenseVector[Double]], DataVectorSerializer)
    kryo.register(classOf[DenseMatrix[Double]], MatrixSerializer)

    kryo.register(classOf[String])
    kryo.register(classOf[Weight[_]])

    kryo.register(classOf[GeneralNetwork[_, _]])
    kryo.register(classOf[SimpleNetwork[_]])

    kryo.register(classOf[AverageLedger])
    kryo.register(classOf[BasicLayer])
    kryo.register(classOf[ConcatLedger])
    kryo.register(classOf[FullTensorLayer])
    kryo.register(classOf[LinearTransformLayer])
    kryo.register(classOf[RBFLayer])
    kryo.register(classOf[RNNLedger])
    kryo.register(classOf[SplitTensorLayer])

    kryo.register(HardSigmoid.getClass)
    kryo.register(HardTanh.getClass)
    kryo.register(HyperbolicTangent.getClass)
    kryo.register(Linear.getClass)
    kryo.register(Rectifier.getClass)
    kryo.register(Sigmoid.getClass)
    kryo.register(Softmax.getClass)
    kryo.register(Softplus.getClass)

    kryo.register(GaussianRBF.getClass)
    kryo.register(InverseQuadRBF.getClass)
    kryo.register(HardGaussianRBF.getClass)

    kryo.register(LeakyReLU.getClass)
    kryo.register(classOf[LedgerGroupLayer])

    kryo.register(classOf[AdaDelta])
    kryo.register(classOf[AdaGrad])
    kryo.register(classOf[StochasticGradientDescent])
    kryo.register(classOf[LedgerAdaDelta])
    kryo.register(classOf[LedgerAdaGrad])
    kryo.register(classOf[LedgerSGD])
  }
}

object DataVectorSerializer extends Serializer[DataVec] {
  override def write(kryo: Kryo, output: Output, t: DataVec): Unit = {
    output.writeInt(t.length)
    t.data.foreach(output.writeDouble)
  }

  override def read(kryo: Kryo, input: Input, aClass: Class[DataVec]): DataVec = {
    val dim = input.readInt()
    val array = (0 until dim).map(_ ⇒ input.readDouble()).toArray
    DenseVector(array)
  }
}

object MatrixSerializer extends Serializer[Matrix] {
  override def write(kryo: Kryo, output: Output, t: Matrix): Unit = {
    output.writeInt(t.rows)
    output.writeInt(t.cols)
    t.data.foreach(output.writeDouble)
  }

  override def read(kryo: Kryo, input: Input, aClass: Class[Matrix]): Matrix = {
    val rows = input.readInt()
    val cols = input.readInt()
    val array = (0 until (rows * cols)).map(_ ⇒ input.readDouble()).toArray
    DenseMatrix.create(rows, cols, array)
  }
}