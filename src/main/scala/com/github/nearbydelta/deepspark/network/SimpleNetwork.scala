package com.github.nearbydelta.deepspark.network

import com.github.nearbydelta.deepspark.data._
import org.apache.spark.rdd.RDD

import scala.collection.parallel.ParSeq

/**
 * Created by bydelta on 15. 10. 17.
 */
class SimpleNetwork[Out]() extends Network[DataVec, Out] {
  final def forward(in: RDD[(Long, DataVec)]): RDD[(Long, DataVec)] = forwardRDD(in)

  def forward(in: DataVec) = forwardSingle(in)

  final def forward(in: ParSeq[DataVec]): ParSeq[DataVec] = {
    forwardSeq(in)
  }

  final def backward(error: Seq[DataVec]): Unit = {
    backwardSeq(error)
  }
}
