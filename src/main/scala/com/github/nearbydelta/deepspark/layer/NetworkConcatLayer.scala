package com.github.nearbydelta.deepspark.layer

import breeze.linalg.DenseVector
import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.network.Network
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.ParSeq

/**
 * __Layer__ for network concatenation.
 * @tparam X Input type of networks.
 */
trait NetworkConcatLayer[X] extends InputLayer[Array[X], DataVec] {
  override val outVecOf: (DataVec) ⇒ DataVec = x ⇒ x
  /** Predefined sequence for Backpropagation **/
  @transient val backprop: ArrayBuffer[(Network[X, DataVec], Int)] = ArrayBuffer()
  /** Sequence of networks to be concatenated. **/
  val networks: ArrayBuffer[Network[X, DataVec]] = ArrayBuffer()

  /**
   * Add new network
   * @param net Network to be added
   * @return self
   */
  def addNetwork(net: Network[X, DataVec]): this.type = {
    networks += net
    backprop += net → NOut
    NOut += net.NOut
    this
  }

  override def apply(in: Array[X]): DataVec = {
    val vectors = in.zip(networks).map {
      case (input, net) ⇒
        net.forward(input)
    }

    DenseVector.vertcat(vectors: _*)
  }

  override def apply(in: RDD[(Long, Array[X])]): RDD[(Long, DataVec)] = {
    val rdds = (0 until networks.size).map({ id ⇒
      val net = networks(id)
      net.forward(in.mapValues(x ⇒ x(id))).mapValues(x ⇒ Map(id → x))
    }).seq

    in.context.union(rdds)
      .reduceByKey((x, y) ⇒ x ++ y)
      .mapValues { x ⇒
      DenseVector.vertcat(x.toSeq.sortBy(_._1).map(_._2): _*)
    }
  }

  @deprecated(message = "NetworkConcatLayer cannot be used inside of RDD, because of its implementation.",
    since = "1.0.0")
  override def backprop(seq: ParSeq[((Array[X], DataVec), DataVec)]): ParSeq[DataVec] =
    throw new IllegalStateException("NetworkConcatLayer cannot be used inside of RDD, because of its implementation.")

  override def backward(error: ParSeq[DataVec]): ParSeq[DataVec] = {
    backprop.par.foreach {
      case (net, from) ⇒
        val to = from + net.NOut
        net backward error.map(_.apply(from until to))
    }

    null
  }

  override def broadcast(sc: SparkContext): Unit = networks.par.foreach(_.broadcast(sc))

  override def forward(in: ParSeq[Array[X]]): ParSeq[DataVec] = {
    val result = (0 until networks.size).par.map { id ⇒
      val net = networks(id)
      net.forward(in.map(x ⇒ x(id))).toIterator
    }.toIndexedSeq

    var i = in.length
    val buf = ArrayBuffer[DataVec]()
    while (i > 0) {
      i -= 1
      val vectors = result.map(_.next()).seq
      buf += DenseVector.vertcat(vectors: _*)
    }
    buf.par
  }

  override def initiateBy(builder: WeightBuilder): this.type = {
    networks.foreach(_.initiateBy(builder))
    this
  }

  override def loss: Double = networks.map(_.loss).sum

  override def read(kryo: Kryo, input: Input): Unit = {
    val size = input.readInt()
    networks.clear()
    backprop.clear()
    NOut = 0

    (0 until size).foreach { _ ⇒
      val net = kryo.readClassAndObject(input).asInstanceOf[Network[X, DataVec]]
      addNetwork(net)
    }

    super.read(kryo, input)
  }

  override def setUpdatable(bool: Boolean): Unit = {
    networks.par.foreach(_.setUpdatable(bool))
    super.setUpdatable(bool)
  }

  override def unbroadcast(): Unit = networks.par.foreach(_.unbroadcast())

  override def write(kryo: Kryo, output: Output): Unit = {
    output.writeInt(networks.size)
    networks.foreach(kryo.writeClassAndObject(output, _))
    super.write(kryo, output)
  }
}
