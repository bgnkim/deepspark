package com.github.nearbydelta.deepspark.word.layer

import breeze.linalg.DenseVector
import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.word.{LedgerBuilder, LedgerModel}

import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.ParSeq

/**
 * __Layer__: Basic, Fully-connected Layer
 *
 */
class ConcatLedger(private var takeRight: Boolean = true)
  extends Ledger[DataVec] {
  override val outVecOf: (DataVec) ⇒ DataVec = x ⇒ x
  private var dimension: Int = 0
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
    val buffer = ArrayBuffer[DataVec]()
    if (takeRight) {
      while (buffer.length + x.length < words) {
        buffer += pad
      }
    }
    val it = x.iterator
    while (it.hasNext) {
      buffer += vectorOf(it.next())
    }
    if (!takeRight) {
      while (buffer.length < words) {
        buffer += pad
      }
    }

    DenseVector.vertcat(buffer: _*)
  }

  override def backprop(seq: ParSeq[((Array[Int], DataVec), DataVec)]): ParSeq[DataVec] = {
    seq.foreach { case ((in, _), err) ⇒
      var index = 0
      if (takeRight) {
        while (index + in.length < words) {
          updateWord(padID, err(index * dimension until (index + 1) * dimension))
          index += 1
        }
      }
      val it = in.iterator
      while (it.hasNext) {
        updateWord(it.next(), err(index * dimension until (index + 1) * dimension))
        index += 1
      }
      if (!takeRight) {
        while (index < words) {
          updateWord(padID, err(index * dimension until (index + 1) * dimension))
          index += 1
        }
      }
    }

    algorithm.update()

    null
  }

  override def read(kryo: Kryo, input: Input): Unit = {
    takeRight = input.readBoolean()
    words = input.readInt()
    super.read(kryo, input)

    dimension = model.dimension
  }

  override def withModel(model: LedgerModel, builder: LedgerBuilder): this.type = {
    require(model.padID != -1, "Concatenate Ledger Layer needs pad.")
    dimension = model.dimension
    if (words > 0) {
      NOut = words * dimension
    }

    super.withModel(model, builder)
  }

  override def write(kryo: Kryo, output: Output): Unit = {
    output.writeBoolean(takeRight)
    output.writeInt(words)
    super.write(kryo, output)
  }
}
