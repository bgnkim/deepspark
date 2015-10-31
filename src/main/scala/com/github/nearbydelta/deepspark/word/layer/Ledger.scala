package com.github.nearbydelta.deepspark.word.layer

import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.layer.InputLayer
import com.github.nearbydelta.deepspark.word.{LedgerAlgorithm, LedgerBuilder, LedgerModel}
import com.github.nearbydelta.deepspark.wordvec.{LedgerNote, LedgerNoteZero}
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
  @transient var model: LedgerModel = _
  var bcModel: Broadcast[LedgerModel] = _
  var delta: LedgerNote = _
  @transient var algorithm: LedgerAlgorithm = _
  @transient var builder: LedgerBuilder = _

  def withModel(model: LedgerModel): this.type = {
    this.model = model
    this
  }

  @deprecated
  override def withInput(in: Int): this.type = this

  @deprecated
  override def withOutput(out: Int): this.type = this

  override def loss: Double = algorithm.loss

  override def update(count: Int): Unit = {
    algorithm.update(count)
  }

  override def write(kryo: Kryo, output: Output): Unit = {
    kryo.writeClassAndObject(output, builder)
    model.write(kryo, output)
    super.write(kryo, output)
  }

  override def read(kryo: Kryo, input: Input): Unit = {
    builder = kryo.readClassAndObject(input).asInstanceOf[LedgerBuilder]
    model = new LedgerModel
    model.read(kryo, input)

    require(model.size > 0)
    initiateBy(builder)
    super.read(kryo, input)
  }

  def initiateBy(builder: LedgerBuilder): this.type = {
    require(model != null, "Set model before initiate by builder!")
    this.builder = builder
    delta = LedgerNoteZero()
    algorithm = builder.getUpdater(model.vectors, delta)

    this
  }

  override def broadcast(sc: SparkContext): Unit = {
    bcModel = sc.broadcast(model)
  }

  override def unbroadcast(): Unit = {
    bcModel.unpersist(blocking = false)
  }

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

  protected def pad =
    if (bcModel != null) vectorOf(bcModel.value.padID)
    else vectorOf(padID)

  protected def vectorOf(str: Int) =
    if (bcModel != null) bcModel.value.vectorAt(str)
    else model.vectorAt(str)
}
