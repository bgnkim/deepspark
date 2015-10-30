package com.github.nearbydelta.deepspark.word.layer

import breeze.linalg.DenseVector
import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.word.LedgerModel

/**
 * __Layer__: Basic, Fully-connected Layer
 *
 */
class AverageLedger extends Ledger[DataVec] {
  override val outVecOf: (DataVec) ⇒ DataVec = x ⇒ x

  override def withModel(model: LedgerModel): this.type = {
    NOut = model.dimension
    super.withModel(model)
  }

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

  override def backward(in: Array[Int], out: DataVec, err: DataVec): DataVec = {
    if (in.nonEmpty) {
      err :/= in.length.toDouble

      in.foreach { str ⇒
        updateWord(str, err)
      }
    } else
      updateWord(padID, err)

    null
  }
}
