package com.github.nearbydelta.deepspark

import com.github.nearbydelta.deepspark.data.{DataVec, Zero}

import scala.collection.mutable
import scala.language.implicitConversions

/**
 * Created by bydelta on 15. 10. 4.
 */
package object wordvec {
  type LedgerNote = mutable.Map[Int, DataVec]
  type ListLedger = IndexedSeq[DataVec]

  implicit object LedgerNoteZero extends Zero[LedgerNote] {
    override def apply(initialValue: LedgerNote): LedgerNote = apply()

    def apply(): LedgerNote = mutable.HashMap()
  }

}
