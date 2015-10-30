package com.github.nearbydelta.deepspark.layer

import breeze.linalg.DenseVector
import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.network.Network
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.ParSeq

/**
 * Created by bydelta on 15. 10. 17.
 */
trait NetworkConcatLayer[X] extends InputLayer[Array[X], DataVec] {
  override val outVecOf: (DataVec) ⇒ DataVec = x ⇒ x
  val networks: ArrayBuffer[Network[X, DataVec]] = ArrayBuffer()
  @transient val backprop: ArrayBuffer[(Network[X, DataVec], Int)] = ArrayBuffer()

  def addNetwork(net: Network[X, DataVec]): this.type = {
    networks += net
    backprop += net → NOut
    withOutput(NOut + net.NOut)
    this
  }

  override def write(kryo: Kryo, output: Output): Unit = {
    output.writeInt(networks.size)
    networks.foreach(kryo.writeClassAndObject(output, _))
    super.write(kryo, output)
  }

  override def read(kryo: Kryo, input: Input): Unit = {
    val size = input.readInt()
    networks.clear()

    (0 until size).foreach { _ ⇒
      val net = kryo.readClassAndObject(input).asInstanceOf[Network[X, DataVec]]
      networks += net
      NOut += net.NOut
      backprop += net → NOut
    }

    super.read(kryo, input)
  }

  override def setUpdatable(bool: Boolean): Unit = {
    networks.foreach(_.setUpdatable(bool))
    super.setUpdatable(bool)
  }

  override def update(count: Int): Unit = networks.par.map { x ⇒ x.update(count); 0 }

  override def apply(in: Array[X]): DataVec = {
    val vectors = in.zip(networks).map {
      case (input, net) ⇒
        net.forward(input)
    }

    DenseVector.vertcat(vectors: _*)
  }

  @deprecated(message = "NetworkConcatLayer cannot be used inside of RDD, because of its implementation.",
    since = "1.0.0")
  override def backward(in: Array[X], out: DataVec, error: DataVec): DataVec =
    throw new IllegalStateException("NetworkConcatLayer cannot be used inside of RDD, because of its implementation.")

  override def loss: Double = networks.map(_.loss).sum

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

  override def forward(in: ParSeq[Array[X]]): ParSeq[DataVec] = {
    (0 until networks.size).par.map { id ⇒
      val net = networks(id)
      net.forward(in.map(x ⇒ x(id)))
    }.seq.foldLeft(ParSeq.empty[Seq[DataVec]]) {
      case (seq, arr) ⇒
        if (seq.isEmpty) arr.map(x ⇒ Seq(x))
        else arr.zip(seq).map {
          case (a, b) ⇒ a +: b
        }
    }.map { x ⇒
      DenseVector.vertcat(x: _*)
    }
  }

  override def backward(error: Seq[DataVec]): Seq[DataVec] = {
    backprop.par.foreach {
      case (net, from) ⇒
        val to = from + net.NOut
        net backward error.map(_.apply(from until to))
    }

    Seq.empty
  }
}
