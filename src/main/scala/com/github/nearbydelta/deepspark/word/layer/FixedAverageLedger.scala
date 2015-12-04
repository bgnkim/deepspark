package com.github.nearbydelta.deepspark.word.layer

import breeze.linalg._
import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.word.LedgerModel

/**
 * __Layer__: Basic, Fully-connected Layer
 *
 */
class FixedAverageLedger extends FixedLedger[DataVec] {
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

  override def withModel(model: LedgerModel): this.type = {
    NOut = model.dimension
    super.withModel(model)
  }
}
