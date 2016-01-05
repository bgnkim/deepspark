package com.github.nearbydelta.deepspark.word.layer

import breeze.linalg.DenseVector
import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.layer.{InputLayer, TransformLayer}
import com.github.nearbydelta.deepspark.word._
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

import scala.reflect.{ClassTag, classTag}

/**
 * __Layer__: Basic, Fully-connected Layer
 *
 */
trait FeatureLedger[OutInfo] extends InputLayer[Array[(Int, DataVec)], OutInfo] {
  @transient implicit override protected val evidenceI: ClassTag[Array[(Int, DataVec)]] = classTag[Array[(Int, DataVec)]]
  @transient var algorithm: LedgerAlgorithm = _
  var bcModel: Broadcast[LedgerModel] = _
  @transient var builder: LedgerBuilder = _
  var dimension: Int = 0
  @transient var model: LedgerModel = _
  var weighting: TransformLayer
  protected var padID = -1
  private var layerBuilder: WeightBuilder = _

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
    else if (bcModel != null) vectorOf(bcModel.value.padID, null)
    else vectorOf(padID, null)

  protected def updateWord(word: Int, dx: DataVec): Unit =
    if (word != -1 && algorithm != null) {
      val vec = algorithm.delta.getOrElseUpdate(word, DenseVector.zeros[Double](dimension))
      vec += dx
    }

  protected def vectorOf(str: Int, feature: DataVec): (Double, DataVec) = {
    val score =
      if (feature == null) weighting(DenseVector.zeros[Double](weighting.NIn))(0)
      else weighting(feature)(0)
    val vec = vectorOf(str)
    (score, vec)
  }

  protected def vectorOf(str: Int): DataVec =
    if (bcModel != null) bcModel.value.vectorAt(str)
    else model.vectorAt(str)

  override def broadcast(sc: SparkContext): Unit = {
    bcModel = sc.broadcast(model)
  }

  override def initiateBy(builder: WeightBuilder): this.type = {
    if (weighting != null) {
      layerBuilder = builder
      weighting.initiateBy(builder)
    }
    this
  }

  override def loss: Double = algorithm.loss

  override def read(kryo: Kryo, input: Input): Unit = {
    builder = kryo.readClassAndObject(input).asInstanceOf[LedgerBuilder]
    val model = new LedgerModel
    model.read(kryo, input)
    weighting = kryo.readClassAndObject(input).asInstanceOf[TransformLayer]
    layerBuilder = kryo.readClassAndObject(input).asInstanceOf[WeightBuilder]
    if (weighting != null)
      weighting.initiateBy(layerBuilder)

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
    kryo.writeClassAndObject(output, weighting)
    kryo.writeClassAndObject(output, layerBuilder)
    super.write(kryo, output)
  }
}
