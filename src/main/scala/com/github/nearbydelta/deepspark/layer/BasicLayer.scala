package com.github.nearbydelta.deepspark.layer

import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.nearbydelta.deepspark.data._

import scala.collection.parallel.ParSeq

/**
 * __Layer__: Basic, Fully-connected Layer
 *
 */
class BasicLayer extends TransformLayer {
  /** Bias **/
  val bias: Weight[DataVec] = new Weight[DataVec]
  /** Weight **/
  val weight: Weight[Matrix] = new Weight[Matrix]
  /** Activation **/
  var act: Activation = HyperbolicTangent

  /**
   * Set the activation function.
   * @param act Activation function.
   * @return self
   */
  def withActivation(act: Activation): this.type = {
    this.act = act
    this
  }

  override def apply(x: DataVec): DataVec = {
    val wx: DataVec = weight.value * x
    val wxb: DataVec = wx + bias.value
    act(wxb)
  }

  override def backward(seq: ParSeq[((DataVec, DataVec), DataVec)]): Seq[DataVec] = {
    val (dX, dW, external) = seq.map { case ((in, out), error) ⇒
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

      (dGdX, dGdW, dGdx)
    }.unzip3

    bias updateBy dX.reduce(_ += _)
    weight updateBy dW.reduce(_ += _)

    external.seq
  }

  override def initiateBy(builder: WeightBuilder): this.type = {
    if (NIn > 0 && NOut > 0) {
      val range = act.initialize(NIn, NOut)
      builder.buildMatrix(weight, NOut, NIn, range)
      builder.buildVector(bias, NOut, range)
    }

    this
  }

  override def loss: Double = weight.loss + bias.loss

  override def read(kryo: Kryo, input: Input): Unit = {
    act = kryo.readClassAndObject(input).asInstanceOf[Activation]
    weight.read(kryo, input)
    bias.read(kryo, input)
    super.read(kryo, input)
  }

  override def update(count: Int): Unit = {
    weight.update(count)
    bias.update(count)
  }

  override def write(kryo: Kryo, output: Output): Unit = {
    kryo.writeClassAndObject(output, act)
    weight.write(kryo, output)
    bias.write(kryo, output)
    super.write(kryo, output)
  }
}