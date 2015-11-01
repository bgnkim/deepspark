package com.github.nearbydelta.deepspark.network

import com.github.nearbydelta.deepspark.data._
import org.apache.spark.rdd.RDD

import scala.collection.parallel.ParSeq

/**
 * __Network__ without input layer. (Input with vector)
 * @tparam Out Type of output.
 */
class SimpleNetwork[Out]() extends Network[DataVec, Out] {
  override def backward(error: Seq[DataVec]): Unit = {
    backwardSeq(error)
  }

  override def forward(in: DataVec) = forwardSingle(in)

  override def forward(in: ParSeq[DataVec]): ParSeq[DataVec] = {
    forwardSeq(in)
  }

  override def forward(in: RDD[(Long, DataVec)]): RDD[(Long, DataVec)] = forwardRDD(in)
}
