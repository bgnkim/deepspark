package com.github.nearbydelta.deepspark.word.layer

import com.github.nearbydelta.deepspark.layer.NetworkConcatLayer

import scala.reflect.{ClassTag, classTag}

/**
 * Created by bydelta on 15. 10. 21.
 */
class LedgerGroupLayer extends NetworkConcatLayer[Array[Int]] {
  @transient implicit override protected val evidenceI: ClassTag[Array[Array[Int]]] = classTag[Array[Array[Int]]]
}
