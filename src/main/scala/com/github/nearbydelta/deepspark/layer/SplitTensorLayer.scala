package com.github.nearbydelta.deepspark.layer

import breeze.linalg.DenseVector
import com.github.nearbydelta.deepspark.data._

/**
 * __Layer__: Basic, Fully-connected Rank 3 Tensor Layer.
 *
 * @note <pre>
 *       v0 = a column vector concatenate v2 after v1 (v11, v12, ... v1in1, v21, ...)
 *       Q = Rank 3 Tensor with size out, in1 × in2 is its entry.
 *       L = Rank 3 Tensor with size out, 1 × (in1 + in2) is its entry.
 *       b = out × 1 matrix.
 *
 *       output = f( v1'.Q.v2 + L.v0 + b )
 *       </pre>
 */
class SplitTensorLayer extends Rank3TensorLayer {

  def withSplit(head: Int, tail: Int): this.type = {
    fanInA = head
    fanInB = tail
    NIn = head + tail
    this
  }

  /**
   * Retrieve first input
   *
   * @param x input to be separated
   * @return first input
   */
  protected override def in1(x: DataVec): DataVec = x(0 until fanInA)

  /**
   * Retrive second input
   * @param x input to be separated
   * @return second input
   */
  protected override def in2(x: DataVec): DataVec = x(fanInA to -1)

  /**
   * Reconstruct error from fragments
   * @param in1 error of input1
   * @param in2 error of input2
   * @return restored error
   */
  override protected def restoreError(in1: DataVec, in2: DataVec): DataVec = DenseVector.vertcat(in1, in2)
}
