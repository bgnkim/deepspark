package com.github.nearbydelta.deepspark.network

import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.layer.InputLayer
import org.apache.spark.rdd.RDD

import scala.collection.parallel.ParSeq

/**
 * Created by bydelta on 15. 10. 17.
 */
class GeneralNetwork[In, Out](var inputLayer: InputLayer[In, _]) extends Network[In, Out] {
  @deprecated(message = "This is for kryo deserialization. Please use this(inputlayer)")
  def this() = this(null)

  override def setUpdatable(bool: Boolean): Network[In, Out] = {
    inputLayer.setUpdatable(bool)
    super.setUpdatable(bool)
  }

  override def NOut: Int =
    layerSeq.lastOption match {
      case Some(x) ⇒ x.NOut
      case None if inputLayer != null ⇒ inputLayer.NOut
      case None ⇒ 0
    }

  override def loss: Double = super.loss + inputLayer.loss

  override def update(count: Int): Unit = {
    super.update(count)
    inputLayer.update(count)
  }

  final def forward(in: RDD[(Long, In)]): RDD[(Long, DataVec)] = {
    val out = inputLayer.forward(in)
    forwardRDD(out)
  }

  def forward(in: In) = {
    val out = inputLayer.forward(in)
    forwardSingle(out)
  }

  final def forward(in: ParSeq[In]): ParSeq[DataVec] = {
    val out = inputLayer.forward(in)
    forwardSeq(out)
  }

  final def backward(error: Seq[DataVec]): Unit = {
    inputLayer backward backwardSeq(error)
  }

  override def write(kryo: Kryo, output: Output): Unit = {
    kryo.writeClassAndObject(output, inputLayer)
    super.write(kryo, output)
  }

  override def read(kryo: Kryo, input: Input): Unit = {
    inputLayer = kryo.readClassAndObject(input).asInstanceOf[InputLayer[In, _]]
    super.read(kryo, input)
  }
}
