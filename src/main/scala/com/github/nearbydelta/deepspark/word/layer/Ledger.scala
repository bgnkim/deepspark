package com.github.nearbydelta.deepspark.word.layer

import breeze.linalg.DenseVector
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
  @transient implicit override protected val evidenceI: ClassTag[Array[Int]] = classTag[Array[Int]]
  @transient var algorithm: LedgerAlgorithm = _
  var bcModel: Broadcast[LedgerModel] = _
  @transient var builder: LedgerBuilder = _
  var dimension: Int = 0
  @transient var model: LedgerModel = _
  protected var padID = -1

  def withModel(model: LedgerModel, builder: LedgerBuilder): this.type = {
    this.model = model
    this.builder = builder
    this.padID = model.padID
    this.dimension = model.dimension
    this.algorithm = builder.getUpdater(this.model.vectors)
    this
  }

  protected def pad =
    if (padID == -1) null
    else if (bcModel != null) vectorOf(bcModel.value.padID)
    else vectorOf(padID)

  protected def updateWord(word: Int, dx: DataVec): Unit =
    if (word != -1 && algorithm != null) {
      val vec = algorithm.delta.getOrElseUpdate(word, DenseVector.zeros[Double](dimension))
      vec += dx
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
    val model = new LedgerModel
    model.read(kryo, input)

    require(model.size > 0, "Model is empty!")
    withModel(model, builder)
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
    kryo.writeClassAndObject(output, builder)
    model.write(kryo, output)
    super.write(kryo, output)
  }
}
