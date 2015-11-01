package com.github.nearbydelta.deepspark.layer

import com.github.nearbydelta.deepspark.data._

/**
 * __Layer__: Basic, Fully-connected Rank 3 Tensor Layer.
 *
 * @note <pre>
 *       v0 = a column vector
 *       Q = Rank 3 Tensor with size out, in × in is its entry.
 *       L = Rank 3 Tensor with size out, 1 × in is its entry.
 *       b = out × 1 matrix.
 *
 *       output = f( v0'.Q.v0 + L.v0 + b )
 *       </pre>
 */
class FullTensorLayer extends Rank3TensorLayer {
  override def withInput(in: Int): this.type = {
    fanInA = NIn
    fanInB = NIn
    super.withInput(in)
  }

  protected override def in1(x: DataVec): DataVec = x

  protected override def in2(x: DataVec): DataVec = x

  protected override def restoreError(in1: DataVec, in2: DataVec): DataVec = in1 + in2
}
