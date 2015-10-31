package com.github.nearbydelta.deepspark.word.layer

import breeze.linalg.DenseVector
import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.word.LedgerModel

import scala.collection.parallel.ParSeq

/**
 * __Layer__: Basic, Fully-connected Layer
 *
 */
class ConcatLedger(private var takeRight: Boolean = true)
  extends Ledger[DataVec] {
  override val outVecOf: (DataVec) ⇒ DataVec = x ⇒ x
  private var words: Int = 0
  private var dimension: Int = 0

  def this() = this(true)

  def withWords(n: Int) = {
    words = n

    if (dimension > 0) {
      NOut = words * dimension
    }
    this
  }

  override def withModel(model: LedgerModel): this.type = {
    dimension = model.dimension
    if (words > 0) {
      NOut = words * dimension
    }

    super.withModel(model)
  }

  override def write(kryo: Kryo, output: Output): Unit = {
    output.writeBoolean(takeRight)
    output.writeInt(words)
    super.write(kryo, output)
  }

  override def read(kryo: Kryo, input: Input): Unit = {
    takeRight = input.readBoolean()
    words = input.readInt()
    super.read(kryo, input)

    dimension = model.dimension
  }

  override def apply(x: Array[Int]): DataVec = {
    val padded =
      if (takeRight) {
        val wordseq = x.takeRight(words).map(vectorOf)
        Seq.fill(words - wordseq.length)(pad) ++ wordseq.toSeq
      } else {
        val wordseq = x.take(words).map(vectorOf)
        wordseq.toSeq ++ Seq.fill(words - wordseq.length)(pad)
      }
    val matrixList = padded
    DenseVector.vertcat(matrixList: _*)
  }

  override def backward(seq: ParSeq[((Array[Int], DataVec), DataVec)]): Seq[DataVec] = {
    seq.foreach { case ((in, _), err) ⇒
      val padded =
        if (takeRight) {
          val wordseq = in.takeRight(words)
          Seq.fill(words - wordseq.length)(padID) ++ wordseq.toSeq
        } else {
          val wordseq = in.take(words)
          wordseq.toSeq ++ Seq.fill(words - wordseq.length)(padID)
        }

      padded.foldLeft(0) {
        case (index, str) ⇒
          val endAt = index + dimension
          val e = err(index until endAt)
          updateWord(str, e)
          endAt
      }
    }

    null
  }
}
