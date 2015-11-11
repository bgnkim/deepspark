package com.github.nearbydelta.deepspark.network

import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.layer.InputLayer
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.parallel.ParSeq

/**
 * __Network__ with general input
 *
 * @param inputLayer Input layer
 * @tparam In Input type
 * @tparam Out Output type
 */
class GeneralNetwork[In, Out](var inputLayer: InputLayer[In, _]) extends Network[In, Out] {
  @deprecated(message = "This is for kryo deserialization. Please use this(inputlayer)")
  def this() = this(null)

  override def NOut: Int =
    layerSeq.lastOption match {
      case Some(x) ⇒ x.NOut
      case None if inputLayer != null ⇒ inputLayer.NOut
      case None ⇒ 0
    }

  override def backward(error: ParSeq[DataVec]): Unit = {
    inputLayer backward backwardSeq(error)
  }

  override def broadcast(sc: SparkContext): Unit = {
    inputLayer.broadcast(sc)
    super.broadcast(sc)
  }

  override def forward(in: In) = {
    val out = inputLayer.forward(in)
    forwardSingle(out)
  }

  override def forward(in: ParSeq[In]): ParSeq[DataVec] = {
    val out = inputLayer.forward(in)
    forwardSeq(out)
  }

  override def forward(in: RDD[(Long, In)]): RDD[(Long, DataVec)] = {
    val out = inputLayer.forward(in)
    broadcast(in.context)
    forwardRDD(out)
  }

  override def initiateBy(builder: WeightBuilder): this.type = {
    inputLayer.initiateBy(builder)
    super.initiateBy(builder)
    this
  }

  override def loss: Double = super.loss + inputLayer.loss

  override def read(kryo: Kryo, input: Input): Unit = {
    inputLayer = kryo.readClassAndObject(input).asInstanceOf[InputLayer[In, _]]
    super.read(kryo, input)
  }

  override def setUpdatable(bool: Boolean): Network[In, Out] = {
    inputLayer.setUpdatable(bool)
    super.setUpdatable(bool)
  }

  override def unbroadcast(): Unit = {
    inputLayer.unbroadcast()
    super.unbroadcast()
  }

  override def write(kryo: Kryo, output: Output): Unit = {
    kryo.writeClassAndObject(output, inputLayer)
    super.write(kryo, output)
  }
}
