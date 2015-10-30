package com.github.nearbydelta.deepspark.word.layer

import breeze.linalg.DenseVector
import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.layer.TransformLayer
import com.github.nearbydelta.deepspark.word.LedgerModel

import scala.annotation.tailrec

/**
 * __Layer__: Basic, Fully-connected Layer
 *
 */
class RNNLedger(var layer: TransformLayer)
  extends Ledger[Array[DataVec]] {
  override val outVecOf: (Array[DataVec]) ⇒ DataVec = _.head
  private var layerBuilder: WeightBuilder = _

  def this() = this(null)

  override def withModel(model: LedgerModel): this.type = {
    NOut = model.dimension
    layer.setUpdatable(false)
    super.withModel(model)
  }

  def initLayerBy(builder: WeightBuilder): this.type = {
    layerBuilder = builder
    layer.initiateBy(builder)
    this
  }

  override def loss: Double = layer.loss + super.loss

  override def write(kryo: Kryo, output: Output): Unit = {
    kryo.writeClassAndObject(output, layerBuilder)
    kryo.writeClassAndObject(output, layer)
    super.write(kryo, output)
  }

  override def read(kryo: Kryo, input: Input): Unit = {
    layerBuilder = kryo.readClassAndObject(input).asInstanceOf[WeightBuilder]
    layer = kryo.readClassAndObject(input).asInstanceOf[TransformLayer]
    layer.initiateBy(layerBuilder)
    super.read(kryo, input)
  }

  override def apply(x: Array[Int]): Array[DataVec] = {
    val seq = x.map(vectorOf)
    if (seq.nonEmpty)
      seq.tail.foldLeft(Seq(seq.head)) {
        case (leftPhrases, curr) ⇒
          val input = DenseVector.vertcat(leftPhrases.head, curr)
          val traceResult = layer.forward(input)
          traceResult +: leftPhrases
      }.toArray
    else
      Array(pad)
  }

  @tailrec
  final def update(error: DataVec, words: Array[Int], in: Array[DataVec]): Unit =
    if (words.length == 1) {
      updateWord(words.head, error)
    } else if (words.length > 1) {
      val curr = words.head
      val o = in.head
      val i1 = in.tail.head
      val i2 = vectorOf(curr)

      // Scaling Down Trick for layer (Pascanu et al., 2013)
      val compoundErr = layer.backward(DenseVector.vertcat(i1, i2), o, error)
      val errorRight = compoundErr(NOut to -1)
      val errorLeft = compoundErr(0 until NOut)

      // Split error term
      updateWord(curr, errorRight)
      update(errorLeft, words.tail, in.tail)
    }

  override def backward(in: Array[Int], out: Array[DataVec], err: DataVec): DataVec = {
    if (in.nonEmpty)
      update(err, in.reverse, out)
    else
      updateWord(padID, err)
    null
  }
}
