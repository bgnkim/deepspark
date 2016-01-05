package com.github.nearbydelta.deepspark.word.layer

import breeze.linalg.DenseVector
import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.layer.TransformLayer
import com.github.nearbydelta.deepspark.word.{LedgerBuilder, LedgerModel}

import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.ParSeq

/**
 * __Layer__: Basic, Fully-connected Layer
 *
 */
@SerialVersionUID(-6414147385334578292L)
class FeatureConcatLedger(override var weighting: TransformLayer,
                          var takeRight: Boolean = true)
  extends FeatureLedger[(DataVec, Array[Double])] {
  override val outVecOf: ((DataVec, Array[Double])) ⇒ DataVec = _._1
  private var words: Int = 0

  def this() = this(null, true)

  def withWords(n: Int) = {
    words = n

    if (dimension > 0) {
      NOut = words * dimension
    }
    this
  }

  override def apply(x: Array[(Int, DataVec)]): (DataVec, Array[Double]) = {
    val vec = DenseVector.zeros[Double](NOut)
    val (ps, pv) = pad
    val buf = Array.fill(words)(ps)

    if (takeRight) {
      x.takeRight(words).foldLeft(0) {
        case (i, (word, feature)) ⇒
          val start = i * dimension
          val end = start + dimension
          val (s, v) = vectorOf(word, feature)
          vec.slice(start, end) := (v * s)
          buf(i) = s
          i + 1
      }
      (x.length until words).par.foreach { x ⇒
        val start = x * dimension
        val end = (x + 1) * dimension

        vec.slice(start, end) := pv
      }
    } else {
      val pads = Math.max(words - x.length, 0)
      (0 until pads).par.foreach { x ⇒
        val start = x * dimension
        val end = (x + 1) * dimension
        vec.slice(start, end) := pv
      }
      x.take(words).foldLeft(pads) {
        case (i, (word, feature)) ⇒
          val start = i * dimension
          val end = start + dimension
          val (s, v) = vectorOf(word, feature)

          vec.slice(start, end) := (v * s)
          buf(i) = s
          i + 1
      }
    }

    (vec, buf)
  }

  override def backprop(seq: ParSeq[((Array[(Int, DataVec)], (DataVec, Array[Double])), DataVec)])
  : (ParSeq[DataVec], ParSeq[() ⇒ Unit]) = {
    val (ps, pv) = pad
    val dvZero = DenseVector.zeros[Double](weighting.NIn)
    val dvScore = DenseVector(ps)

    val buf = seq.flatMap { case ((in, out), err) ⇒
      val buf = ArrayBuffer[((DataVec, DataVec), DataVec)]()
      val scores = out._2
      if (takeRight) {
        in.takeRight(words).foldLeft(0) {
          case (i, (word, feature)) ⇒
            val start = i * dimension
            val end = start + dimension
            val e = err.slice(start, end)

            val score = scores(i)
            val vec = vectorOf(word)

            buf append ((feature, DenseVector(score)) → DenseVector(e dot vec))
            updateWord(word, e :* score)
            i + 1
        }
        (in.length until words).foreach { x ⇒
          val start = x * dimension
          val end = (x + 1) * dimension
          val e = err.slice(start, end)

          buf append ((dvZero, dvScore) → DenseVector(e dot pv))
          updateWord(padID, e :* ps)
        }
      } else {
        val pads = Math.max(words - in.length, 0)
        (0 until pads).foreach { x ⇒
          val start = x * dimension
          val end = (x + 1) * dimension
          val e = err.slice(start, end)

          buf append ((dvZero, dvScore) → DenseVector(e dot pv))
          updateWord(padID, e :* ps)
        }
        in.take(words).foldLeft(pads) {
          case (i, (word, feature)) ⇒
            val start = i * dimension
            val end = start + dimension
            val e = err.slice(start, end)

            val score = scores(i)
            val vec = vectorOf(word)

            buf append ((feature, DenseVector(score)) → DenseVector(e dot vec))
            updateWord(word, e :* score)
            i + 1
        }
      }
      buf
    }

    val (_, f) = weighting.backprop(buf)

    (null, ParSeq(algorithm.update) ++ f)
  }

  override def read(kryo: Kryo, input: Input): Unit = {
    takeRight = input.readBoolean()
    words = input.readInt()
    super.read(kryo, input)
  }

  override def withModel(model: LedgerModel, builder: LedgerBuilder): this.type = {
    require(model.padID != -1, "Concatenate Ledger Layer needs pad.")
    if (words > 0) {
      NOut = words * model.dimension
    }

    super.withModel(model, builder)
  }

  override def write(kryo: Kryo, output: Output): Unit = {
    output.writeBoolean(takeRight)
    output.writeInt(words)
    super.write(kryo, output)
  }
}
