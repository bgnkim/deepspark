package com.github.nearbydelta.deepspark.word.layer

import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.layer.InputLayer
import com.github.nearbydelta.deepspark.word._
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

import scala.reflect.{ClassTag, classTag}

/**
 * __Layer__: Basic, Fully-connected Layer
 *
 */
trait Ledger[OutInfo] extends InputLayer[Array[Int], OutInfo] {
  protected lazy val padID = model.padID
  @transient implicit override protected val evidenceI: ClassTag[Array[Int]] = classTag[Array[Int]]
  @transient var algorithm: LedgerAlgorithm = _
  var bcModel: Broadcast[LedgerModel] = _
  @transient var builder: LedgerBuilder = _
  var delta: LedgerNote = _
  @transient var model: LedgerModel = _

  def initiateBy(builder: LedgerBuilder): this.type = {
    require(model != null, "Set model before initiate by builder!")
    this.builder = builder
    delta = LedgerNoteZero()
    algorithm = builder.getUpdater(model.vectors, delta)

    this
  }

  def withModel(model: LedgerModel): this.type = {
    this.model = model
    this
  }

  protected def pad =
    if (bcModel != null) vectorOf(bcModel.value.padID)
    else vectorOf(padID)

  protected def updateWord(word: Int, vec: DataVec): Unit = {
    val dx = Weight.scaleCheck(vec)
    delta.synchronized {
      delta.get(word) match {
        case Some(x) ⇒ x += dx
        case None ⇒
          delta(word) = dx
      }
    }
  }

  protected def vectorOf(str: Int) =
    if (bcModel != null) bcModel.value.vectorAt(str)
    else model.vectorAt(str)

  override def broadcast(sc: SparkContext): Unit = {
    bcModel = sc.broadcast(model)
  }

  override def loss: Double = algorithm.loss

  override def read(kryo: Kryo, input: Input): Unit = {
    builder = kryo.readClassAndObject(input).asInstanceOf[LedgerBuilder]
    model = new LedgerModel
    model.read(kryo, input)

    require(model.size > 0)
    initiateBy(builder)
    super.read(kryo, input)
  }

  override def unbroadcast(): Unit = {
    bcModel.unpersist(blocking = false)
  }

  override def update(count: Int): Unit = {
    algorithm.update(count)
  }

  @deprecated
  override def withInput(in: Int): this.type = this

  @deprecated
  override def withOutput(out: Int): this.type = this

  override def write(kryo: Kryo, output: Output): Unit = {
    kryo.writeClassAndObject(output, builder)
    model.write(kryo, output)
    super.write(kryo, output)
  }
}
