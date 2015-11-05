package com.github.nearbydelta.deepspark.word.layer

import breeze.linalg.DenseVector
import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.word.LedgerModel

import scala.collection.parallel.ParSeq

/**
 * __Layer__: Basic, Fully-connected Layer
 *
 */
class FixedAverageLedger extends FixedLedger[DataVec] {
  override val outVecOf: (DataVec) ⇒ DataVec = x ⇒ x

  override def apply(x: Array[Int]): DataVec = {
    if (x.nonEmpty) {
      val matrix =
        x.foldLeft(DenseVector.zeros[Double](NOut)) {
          case (acc, str) ⇒ acc += vectorOf(str)
        }
      matrix :/= x.length.toDouble
    } else
      pad
  }

  override def backward(seq: ParSeq[((Array[Int], DataVec), DataVec)]): Seq[DataVec] = null

  override def withModel(model: LedgerModel): this.type = {
    NOut = model.dimension
    super.withModel(model)
  }
}
