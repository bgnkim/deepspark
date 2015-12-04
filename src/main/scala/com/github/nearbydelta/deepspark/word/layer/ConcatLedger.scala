package com.github.nearbydelta.deepspark.word.layer

import breeze.linalg.DenseVector
import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.word.{LedgerBuilder, LedgerModel}

import scala.collection.parallel.ParSeq

/**
 * __Layer__: Basic, Fully-connected Layer
 *
 */
class ConcatLedger(private var takeRight: Boolean = true)
  extends Ledger[DataVec] {
  override val outVecOf: (DataVec) ⇒ DataVec = x ⇒ x
  private var words: Int = 0

  def this() = this(true)

  def withWords(n: Int) = {
    words = n

    if (dimension > 0) {
      NOut = words * dimension
    }
    this
  }

  override def apply(x: Array[Int]): DataVec = {
    val vec = DenseVector.zeros[Double](NOut)
    if (takeRight) {
      x.takeRight(words).foldLeft(0) {
        case (start, word) ⇒
          val end = start + dimension
          vec.slice(start, end) := vectorOf(word)
          end
      }
      (x.length until words).par.foreach { x ⇒
        val start = x * dimension
        val end = (x + 1) * dimension
        vec.slice(start, end) := pad
      }
    } else {
      val pads = Math.max(words - x.length, 0)
      (0 until pads).par.foreach { x ⇒
        val start = x * dimension
        val end = (x + 1) * dimension
        vec.slice(start, end) := pad
      }
      x.take(words).foldLeft(pads * dimension) {
        case (start, word) ⇒
          val end = start + dimension
          vec.slice(start, end) := vectorOf(word)
          end
      }
    }

    vec
  }

  override def backprop(seq: ParSeq[((Array[Int], DataVec), DataVec)]): (ParSeq[DataVec], ParSeq[() ⇒ Unit]) = {
    seq.foreach { case ((in, _), err) ⇒
      if (takeRight) {
        in.takeRight(words).foldLeft(0) {
          case (start, word) ⇒
            val end = start + dimension
            updateWord(word, err.slice(start, end))
            end
        }
        (in.length until words).foreach { x ⇒
          val start = x * dimension
          val end = (x + 1) * dimension
          updateWord(padID, err.slice(start, end))
        }
      } else {
        val pads = Math.max(words - in.length, 0)
        (0 until pads).foreach { x ⇒
          val start = x * dimension
          val end = (x + 1) * dimension
          updateWord(padID, err.slice(start, end))
        }
        in.take(words).foldLeft(pads * dimension) {
          case (start, word) ⇒
            val end = start + dimension
            updateWord(word, err.slice(start, end))
            end
        }
      }
    }

    (null, ParSeq(algorithm.update))
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
