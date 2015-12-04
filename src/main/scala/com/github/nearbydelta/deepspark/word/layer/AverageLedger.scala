package com.github.nearbydelta.deepspark.word.layer

import breeze.linalg.{DenseVector, axpy}
import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.word.{LedgerBuilder, LedgerModel}

import scala.collection.parallel.ParSeq

/**
 * __Layer__: Basic, Fully-connected Layer
 *
 */
class AverageLedger extends Ledger[DataVec] {
  override val outVecOf: (DataVec) ⇒ DataVec = x ⇒ x

  override def apply(x: Array[Int]): DataVec = {
    if (x.nonEmpty) {
      val matrix = DenseVector.zeros[Double](NOut)
      val it = x.toIterator
      val factor = 1.0 / x.length
      while (it.hasNext) {
        axpy(factor, vectorOf(it.next()), matrix)
      }
      matrix
    } else
      pad
  }

  override def backprop(seq: ParSeq[((Array[Int], DataVec), DataVec)]): (ParSeq[DataVec], ParSeq[() ⇒ Unit]) = {
    seq.foreach { case ((in, _), err) ⇒
      if (in.nonEmpty) {
        err :/= in.length.toDouble
        val it = in.iterator
        while (it.hasNext) {
          updateWord(it.next(), err)
        }
      } else
        updateWord(padID, err)
    }

    (null, ParSeq(algorithm.update))
  }

  override def withModel(model: LedgerModel, builder: LedgerBuilder): this.type = {
    NOut = model.dimension
    super.withModel(model, builder)
  }
}
