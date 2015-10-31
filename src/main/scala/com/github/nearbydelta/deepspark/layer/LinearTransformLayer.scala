package com.github.nearbydelta.deepspark.layer

import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.nearbydelta.deepspark.data._

import scala.collection.parallel.ParSeq

/**
 * __Layer__: Basic, Fully-connected Layer
 *
 */
class LinearTransformLayer extends TransformLayer {
  /* Initialize weight */
  val weight: Weight[Matrix] = new Weight[Matrix]
  var act: Activation = HyperbolicTangent

  def withActivation(act: Activation): this.type = {
    this.act = act
    this
  }

  override def initiateBy(builder: WeightBuilder): this.type = {
    if (NIn > 0 && NOut > 0) {
      val range = act.initialize(NIn, NOut)
      builder.buildMatrix(weight, NOut, NIn, range)
    }

    this
  }

  override def loss: Double = weight.loss

  override def write(kryo: Kryo, output: Output): Unit = {
    kryo.writeClassAndObject(output, act)
    weight.write(kryo, output)
    super.write(kryo, output)
  }

  override def read(kryo: Kryo, input: Input): Unit = {
    act = kryo.readClassAndObject(input).asInstanceOf[Activation]
    weight.read(kryo, input)
    super.read(kryo, input)
  }

  override def update(count: Int): Unit = {
    weight.update(count)
  }

  /**
   * Forward computation
   *
   * @param x input matrix
   * @return output matrix
   */
  override def apply(x: DataVec): DataVec = {
    val wx: DataVec = weight.value * x
    act(wx)
  }

  /**
   * <p>Backward computation.</p>
   *
   * @note <p>
   *       Let this layer have function F composed with function <code> X(x) = W.x + b </code>
   *       and higher layer have function G.
   *       </p>
   *
   *       <p>
   *       Weight is updated with: <code>dG/dW</code>
   *       and propagate <code>dG/dx</code>
   *       </p>
   *
   *       <p>
   *       For the computation, we only used denominator layout. (cf. Wikipedia Page of Matrix Computation)
   *       For the computation rules, see "Matrix Cookbook" from MIT.
   *       </p>
   *
   * @return propagated error (in this case, <code>dG/dx</code> )
   */
  def backward(seq: ParSeq[((DataVec, DataVec), DataVec)]): Seq[DataVec] = {
    val (dW, external) = seq.map { case ((in, out), error) ⇒
      val dFdX = act.diffAtY(out)
      /*
       * Chain Rule : dG/dX_ij = tr[ ( dG/dF ).t * dF/dX_ij ].
       *
       * Note 1. X, dG/dF, dF/dX_ij are row vectors. Therefore tr(.) can be omitted.
       *
       * Thus, dG/dX = [ (dG/dF).t * dF/dX ].t, because [...] is 1 × fanOut matrix.
       * Therefore dG/dX = dF/dX * dG/dF, because dF/dX is symmetric in our case.
       */
      val dGdX: DataVec = dFdX * error

      /*
       * Chain Rule : dG/dW_ij = tr[ ( dG/dX ).t * dX/dW_ij ].
       *
       * dX/dW_ij is a fan-Out dimension column vector with all zero but (i, 1) = X_j.
       * Thus, tr(.) can be omitted, and dG/dW_ij = (dX/dW_ij).t * dG/dX
       * Then {j-th column of dG/dW} = X_j * dG/dX = dG/dX * X_j.
       *
       * Therefore dG/dW = dG/dX * X.t
       */
      val dGdW: Matrix = dGdX * in.t

      /*
       * Chain Rule : dG/dx_ij = tr[ ( dG/dX ).t * dX/dx_ij ].
       *
       * X is column vector. Thus j is always 1, so dX/dx_i is a W_?i.
       * Hence dG/dx_i = tr[ (dG/dX).t * dX/dx_ij ] = (W_?i).t * dG/dX.
       *
       * Thus dG/dx = W.t * dG/dX
       */
      val dGdx: DataVec = weight.value.t * dGdX

      (dGdW, dGdx)
    }.unzip

    weight updateBy dW.reduce(_ += _)

    external.seq
  }
}