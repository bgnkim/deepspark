package com.github.nearbydelta.deepspark.word.layer

import breeze.linalg.DenseVector
import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.layer.TransformLayer
import com.github.nearbydelta.deepspark.word.{LedgerBuilder, LedgerModel}

import scala.collection.mutable
import scala.collection.parallel.ParSeq

/**
 * __Layer__: Basic, Fully-connected Layer
 *
 */
class RNNLedger(var layer: TransformLayer)
  extends Ledger[mutable.Stack[DataVec]] {
  override val outVecOf: (mutable.Stack[DataVec]) ⇒ DataVec = _.top
  private var layerBuilder: WeightBuilder = _

  def this() = this(null)

  override def apply(x: Array[Int]): mutable.Stack[DataVec] = {
    val seq = x.map(vectorOf).toIterator
    if (seq.nonEmpty) {
      val buffer = mutable.Stack[DataVec]()
      buffer.push(seq.next())
      while (seq.hasNext) {
        val leftPhrases = buffer.top
        val curr = seq.next()
        val input = DenseVector.vertcat(leftPhrases, curr)
        val traceResult = layer.forward(input)
        buffer.push(input)
        buffer.push(traceResult)
      }

      buffer
    } else
      mutable.Stack(pad)
  }

  override def backprop(seq: ParSeq[((Array[Int], mutable.Stack[DataVec]), DataVec)]): ParSeq[DataVec] = {
    seq.foreach { case ((in, out), err) ⇒
      if (in.nonEmpty) {
        val words = in.reverseIterator
        var error = err

        while (words.hasNext) {
          val o = out.pop()
          val curr = words.next()

          if (out.nonEmpty) {
            val i = out.pop()

            val compoundErr = layer.backprop(ParSeq((i, o) → error)).head
            val errorRight = compoundErr(NOut to -1)
            error = compoundErr(0 until NOut)

            updateWord(curr, errorRight)
          } else {
            updateWord(curr, error)
          }
        }
      } else
        updateWord(padID, err)
    }

    algorithm.update()

    null
  }

  override def initiateBy(builder: WeightBuilder): this.type = {
    layerBuilder = builder
    layer.initiateBy(builder)
    this
  }

  override def loss: Double = layer.loss + super.loss

  override def read(kryo: Kryo, input: Input): Unit = {
    layerBuilder = kryo.readClassAndObject(input).asInstanceOf[WeightBuilder]
    layer = kryo.readClassAndObject(input).asInstanceOf[TransformLayer]
    layer.initiateBy(layerBuilder)
    super.read(kryo, input)
  }

  override def withModel(model: LedgerModel, builder: LedgerBuilder): this.type = {
    NOut = model.dimension
    layer.withInput(model.dimension * 2)
    layer.withOutput(model.dimension)
    layer.setUpdatable(false)
    super.withModel(model, builder)
  }

  override def write(kryo: Kryo, output: Output): Unit = {
    kryo.writeClassAndObject(output, layerBuilder)
    kryo.writeClassAndObject(output, layer)
    super.write(kryo, output)
  }
}
