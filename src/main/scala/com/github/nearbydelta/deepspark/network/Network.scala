package com.github.nearbydelta.deepspark.network

import _root_.java.io.{File, FileInputStream, FileOutputStream}

import com.esotericsoftware.kryo.KryoSerializable
import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.layer.{Layer, TransformLayer}
import com.twitter.chill._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.annotation.tailrec
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.ParSeq
import scala.reflect.ClassTag


/**
 * __Trait__ of network.
 * @tparam In Type of input
 * @tparam Out Type of output.
 */
trait Network[In, Out] extends Serializable with KryoSerializable {
  /** Seqeunce of hidden/output layer to be applied. **/
  protected val layerSeq: mutable.ListBuffer[Layer[DataVec, DataVec]] = mutable.ListBuffer()
  /** Weight builder for this network. **/
  @transient protected var builder: WeightBuilder = _

  /**
   * Size of output
   * @return Output size
   */
  def NOut =
    if (layerSeq.nonEmpty) layerSeq.last.NOut
    else 0

  /**
   * Add new top layer.
   * @param layer Layer to be added on top of current stack.
   * @return self
   */
  final def add(layer: Layer[DataVec, DataVec]): this.type = {
    val prevout = NOut
    if (prevout != 0 && layer.NIn != prevout) {
      layer.withInput(prevout)
    }
    layerSeq += layer
    this
  }

  /**
   * Execute error backpropagation
   * @param err Sequence of error to be backprop.
   */
  def backward(err: ParSeq[DataVec]): ArrayBuffer[() ⇒ Unit]

  /**
   * Broadcast resources of input layer.
   * @param sc Spark Context
   */
  def broadcast(sc: SparkContext): Unit = {}

  /**
   * Apply this network.
   * @param in RDD of (ID, Value).
   * @return RDD of (ID, Output vector).
   */
  def forward(in: RDD[(Long, In)]): RDD[(Long, DataVec)]

  /**
   * Apply this network.
   * @param in Parallel Sequence of input
   * @return Sequence of Output vector.
   */
  def forward(in: ParSeq[In]): ParSeq[DataVec]

  /**
   * Apply this network.
   * @param in input
   * @return Output vector.
   */
  def forward(in: In): DataVec

  /**
   * Initiate this network by given builder
   * @param builder Weight builder
   * @return self
   */
  def initiateBy(builder: WeightBuilder): this.type = {
    this.builder = builder
    layerSeq.foreach(_.initiateBy(builder))
    this
  }

  /**
   * Sequence of hidden/output layers
   * @return Sequence of hidden/output layer
   */
  def layers: Seq[Layer[DataVec, DataVec]] = layerSeq

  /**
   * Weight loss of layers
   * @return Weight loss
   */
  def loss = layerSeq.par.map(_.loss).sum

  /**
   * Apply this network.
   * @param in RDD of Value.
   * @return RDD of Output vector.
   */
  def predictSoft(in: RDD[In])(implicit evidence$1: ClassTag[DataVec]): RDD[DataVec] = forward(in.map(0l → _)).map(_._2)

  /**
   * Apply this network.
   * @param in input
   * @return Output vector.
   */
  def predictSoft(in: In): DataVec = forward(in)

  /**
   * Save this network.
   * @param file Save path of this network.
   */
  def saveTo(file: String) =
    try {
      val output = new Output(new FileOutputStream(new File(file)))
      KryoWrap.get.kryo.writeClassAndObject(output, this)
      output.close()
    } catch {
      case _: Throwable ⇒
    }

  /**
   * Set this network can be used in backpropagation.
   * @param bool True if this should prepare backpropagation
   * @return self
   */
  def setUpdatable(bool: Boolean) = {
    layerSeq.par.foreach(_.setUpdatable(bool))
    this
  }

  /**
   * Unpersist resources of input layer.
   */
  def unbroadcast(): Unit = {}

  /**
   * Backpropagate errors
   * @param err Top-level error to be propagated backward.
   * @return lowest-level error sequence.
   */
  protected final def backwardSeq(err: ParSeq[DataVec]): (ParSeq[DataVec], ArrayBuffer[() ⇒ Unit]) = {
    val it = layerSeq.reverseIterator
    val futures = ArrayBuffer[() ⇒ Unit]()
    var seq = err
    while (it.hasNext) {
      val (s, f) = it.next().backward(seq)
      futures ++= f.seq
      seq = s
    }

    (seq, futures)
  }

  /**
   * Apply by layers.
   * @param in RDD of (ID, Vector)
   * @param layers Sequence of layers to be applied. Default value is layer sequence.
   * @return RDD of final output (ID, Output Vector)
   */
  @tailrec
  protected final def forwardRDD(in: RDD[(Long, DataVec)],
                                 layers: Seq[Layer[DataVec, DataVec]] = layerSeq): RDD[(Long, DataVec)] =
    if (layers.isEmpty) in
    else forwardRDD(layers.head forward in, layers.tail)

  /**
   * Apply by layers.
   * @param in Parallel sequence of Vector
   * @return Parallel sequence of final output
   */
  protected final def forwardSeq(in: ParSeq[DataVec]): ParSeq[DataVec] = {
    val it = layerSeq.toIterator
    var seq = in
    while (it.hasNext) {
      seq = it.next().forward(seq)
    }
    seq
  }

  /**
   * Apply by layers.
   * @param in Vector
   * @param layers Sequence of layers to be applied. Default value is layer sequence.
   * @return Final output Output Vector
   */
  @tailrec
  protected final def forwardSingle(in: DataVec,
                                    layers: Seq[Layer[DataVec, DataVec]] = layerSeq): DataVec =
    if (layers.isEmpty) in
    else forwardSingle(layers.head forward in, layers.tail)

  override def read(kryo: Kryo, input: Input): Unit = {
    builder = kryo.readClassAndObject(input).asInstanceOf[WeightBuilder]
    val size = input.readInt()
    layerSeq.clear()
    (0 until size).foreach { _ ⇒
      val layer = kryo.readClassAndObject(input).asInstanceOf[TransformLayer]
      layer.initiateBy(builder)
      layerSeq append layer
    }
  }

  override def write(kryo: Kryo, output: Output): Unit = {
    kryo.writeClassAndObject(output, builder)
    output.writeInt(layerSeq.length)
    layerSeq.foreach(kryo.writeClassAndObject(output, _))
  }
}

/**
 * Companion object of network.
 */
object Network {
  /**
   * Read network from given file.
   * @param file Path to be read.
   * @tparam T Type of network.
   * @return Restored network.
   */
  def readFrom[T <: Network[_, _]](file: String) = {
    val input = new Input(new FileInputStream(file))
    val obj = KryoWrap.get.kryo.readClassAndObject(input).asInstanceOf[T]
    input.close()
    obj
  }
}