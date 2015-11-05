package com.github.nearbydelta.deepspark.word.layer

import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.nearbydelta.deepspark.layer.InputLayer
import com.github.nearbydelta.deepspark.word._
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

import scala.reflect.{ClassTag, classTag}

/**
 * __Layer__: Basic, Fully-connected Layer
 *
 */
trait FixedLedger[OutInfo] extends InputLayer[Array[Int], OutInfo] {
  @transient implicit override protected val evidenceI: ClassTag[Array[Int]] = classTag[Array[Int]]
  var bcModel: Broadcast[LedgerModel] = _
  @transient var model: LedgerModel = _
  protected var padID = -1

  def withModel(model: LedgerModel): this.type = {
    this.model = model
    this.padID = model.padID
    this
  }

  protected def pad =
    if (padID == -1) null
    else if (bcModel != null) vectorOf(bcModel.value.padID)
    else vectorOf(padID)

  protected def vectorOf(str: Int) =
    if (bcModel != null) bcModel.value.vectorAt(str)
    else model.vectorAt(str)

  override def broadcast(sc: SparkContext): Unit = {
    bcModel = sc.broadcast(model)
  }

  override def loss: Double = 0.0

  override def read(kryo: Kryo, input: Input): Unit = {
    val model = new LedgerModel
    model.read(kryo, input)
    withModel(model)
    super.read(kryo, input)
  }

  override def unbroadcast(): Unit = {
    bcModel.unpersist(blocking = false)
  }

  @deprecated
  override def withInput(in: Int): this.type = this

  @deprecated
  override def withOutput(out: Int): this.type = this

  override def write(kryo: Kryo, output: Output): Unit = {
    model.write(kryo, output)
    super.write(kryo, output)
  }
}
