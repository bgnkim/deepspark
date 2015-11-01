package com.github.nearbydelta.deepspark

import com.github.nearbydelta.deepspark.data.{DataVec, Zero}

import scala.collection.mutable
import scala.language.implicitConversions

/**
 * Package: deepspark.word
 *
 * Package includes related package for word
 */
package object word {
  /** Type: Update store of word list, or ledger */
  type LedgerNote = mutable.Map[Int, DataVec]
  /** Type: Word list, or ledger */
  type ListLedger = IndexedSeq[DataVec]

  /**
   * Zero object for Ledger Note.
   */
  implicit object LedgerNoteZero extends Zero[LedgerNote] {
    def apply(): LedgerNote = mutable.HashMap()

    override def apply(initialValue: LedgerNote): LedgerNote = apply()
  }

}
