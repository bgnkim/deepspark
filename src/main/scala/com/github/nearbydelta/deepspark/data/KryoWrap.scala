package com.github.nearbydelta.deepspark.data

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import breeze.linalg.{DenseMatrix, DenseVector}
import com.esotericsoftware.kryo.Serializer
import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.nearbydelta.deepspark.layer._
import com.github.nearbydelta.deepspark.network.{GeneralNetwork, SimpleNetwork}
import com.github.nearbydelta.deepspark.word._
import com.github.nearbydelta.deepspark.word.layer._
import com.twitter.chill.{Kryo, ScalaKryoInstantiator}
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * Kryo wrapper, which all classes in DeepSpark library are registered.
 */
object KryoWrap extends Serializable{
  /**
   * Kryo object
   */
  val customRegister = ArrayBuffer[Kryo ⇒ Unit]()

  def addRegisterFunction(fn: Kryo ⇒ Unit): Unit = {
    customRegister += fn
  }

  /**
   * Register all classes in DeepSpark library into given Kryo object.
   * @param kryo Kryo object to register classes in DeepSpark library.
   */
  def init(kryo: Kryo): Unit = {
    // Please, preserve this ordering.
    kryo.register(classOf[LedgerWords])
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

    kryo.register(SoftmaxCEE.getClass)
    kryo.register(classOf[FixedAverageLedger])

    customRegister.foreach(fn ⇒ fn(kryo))
  }

  def kryo = {
    val k = new ScalaKryoInstantiator().newKryo()
    init(k)
    k
  }

  def readKryoFile[X](sc: SparkContext, path: String)(implicit evidence$1: ClassTag[X]): RDD[X] = {
    sc.sequenceFile(path, classOf[NullWritable], classOf[BytesWritable], 2)
      .mapPartitions{
      lazy val kryo = KryoWrap.kryo
      _.flatMap{
        case (_, bytes) ⇒
          val bis = new ByteArrayInputStream(bytes.getBytes)
          val input = new Input(bis)
          kryo.readClassAndObject(input).asInstanceOf[Array[X]]
      }
    }
  }

  def saveAsKryoFile[X](rdd: RDD[X], path: String)
                       (implicit evidence$1: ClassTag[X], evidence$2: ClassTag[Array[X]]) = {
    rdd.mapPartitions{
      lazy val kryo = KryoWrap.kryo

      iter ⇒
        iter.grouped[X](10).map{ x ⇒
          val arr = x.toArray[X]
          val bos = new ByteArrayOutputStream()
          val out = new Output(bos, 1024*1024)
          kryo.writeClassAndObject(out, arr)
          out.flush()

          (NullWritable.get(), new BytesWritable(bos.toByteArray))
        }
    }.saveAsSequenceFile(path)
  }
}

/**
 * Customized serializer for Vector type.
 */
object DataVectorSerializer extends Serializer[DataVec] {
  override def read(kryo: Kryo, input: Input, aClass: Class[DataVec]): DataVec = {
    val dim = input.readInt()
    val array = (0 until dim).map(_ ⇒ input.readDouble()).toArray
    DenseVector(array)
  }

  override def write(kryo: Kryo, output: Output, t: DataVec): Unit = {
    output.writeInt(t.length)
    t.data.foreach(output.writeDouble)
  }
}

/**
 * Customized serializer for Matrix type.
 */
object MatrixSerializer extends Serializer[Matrix] {
  override def read(kryo: Kryo, input: Input, aClass: Class[Matrix]): Matrix = {
    val rows = input.readInt()
    val cols = input.readInt()
    val array = (0 until (rows * cols)).map(_ ⇒ input.readDouble()).toArray
    DenseMatrix.create(rows, cols, array)
  }

  override def write(kryo: Kryo, output: Output, t: Matrix): Unit = {
    output.writeInt(t.rows)
    output.writeInt(t.cols)
    t.data.foreach(output.writeDouble)
  }
}