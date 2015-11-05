package com.github.nearbydelta.deepspark.layer

import breeze.linalg.DenseVector
import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.nearbydelta.deepspark.data._

import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.ParSeq

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
abstract class Rank3TensorLayer extends TransformLayer {
  val bias = new Weight[DataVec]
  val linear = new Weight[Matrix]
  /* Initialize weight */
  val quadratic = ArrayBuffer[Weight[Matrix]]()
  var act: Activation = HyperbolicTangent
  /* Number of Fan-ins */
  protected final var fanInA = 0
  protected final var fanInB = 0

  /**
   * Set the activation function.
   * @param act Activation function.
   * @return self
   */
  def withActivation(act: Activation): this.type = {
    this.act = act
    this
  }

  /**
   * Retrieve first input
   *
   * @param x input to be separated
   * @return first input
   */
  protected def in1(x: DataVec): DataVec

  /**
   * Retrive second input
   *
   * @param x input to be separated
   * @return second input
   */
  protected def in2(x: DataVec): DataVec

  /**
   * Reconstruct error from fragments
   * @param in1 error of input1
   * @param in2 error of input2
   * @return restored error
   */
  protected def restoreError(in1: DataVec, in2: DataVec): DataVec

  override def apply(x: DataVec): DataVec = {
    val inA = in1(x)
    val inB = in2(x)

    val intermediate: DataVec = linear.value * x
    intermediate += bias.value

    val quads = quadratic.map { q ⇒
      val Qy: DataVec = q.value * inB
      inA dot Qy
    }
    intermediate += DenseVector(quads: _*)

    act(intermediate)
  }

  override def backward(seq: ParSeq[((DataVec, DataVec), DataVec)]): Seq[DataVec] = {
    val (internal, external) = seq.map { case ((in, out), error) ⇒
      val inA = in1(in)
      val inB = in2(in)
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
     * Chain Rule (Linear weight case) : dG/dW_ij = tr[ ( dG/dX ).t * dX/dW_ij ].
     *
     * dX/dW_ij is a fan-Out dimension column vector with all zero but (i, 1) = X_j.
     * Thus, tr(.) can be omitted, and dG/dW_ij = (dX/dW_ij).t * dG/dX
     * Then {j-th column of dG/dW} = X_j * dG/dX = dG/dX * X_j.
     *
     * Therefore dG/dW = dG/dX * X.t
     */
      val dGdL = dGdX * in.t

      /*
     * Chain Rule (Linear weight part) : dG/dx_ij = tr[ ( dG/dX ).t * dX/dx_ij ].
     *
     * X is column vector. Thus j is always 1, so dX/dx_i is a W_?i.
     * Hence dG/dx_i = tr[ (dG/dX).t * dX/dx_ij ] = (W_?i).t * dG/dX.
     *
     * Thus dG/dx (linear part) = W.t * dG/dX.
     */
      val dGdx = linear.value.t * dGdX

      /*
     * Because X = inA.t * Q * inB, dX/dQ = inA * inB.t
     */
      val dXdQ: Matrix = inA * inB.t //d tr(axb)/dx = a'b'

      // Add dG/dx quadratic part.
      val dGdQ = (0 until NOut).map { id ⇒
        // This is scalar
        val dGdXi = dGdX(id)

        /*
         * Chain Rule (Quadratic weight case) : dG/dQ_ij = tr[ ( dG/dX ).t * dX/dQ_ij ].
         *
         * dX/dQ_ij = (inA * inB.t)_ij, and so dG/dQ_ij = (dG/dX).t * dX/dQ_ij.
         * They are scalar, so dG/dQ = dG/dX * dX/dQ.
         */
        val dGdQ: Matrix = dXdQ :* dGdXi

        /*
         * Chain Rule (Linear weight part) : dG/dx_ij = tr[ ( dG/dX ).t * dX/dx_ij ].
         *
         * X is column vector. Thus j is always 1, so dX/dx_i is a W_?i.
         * Hence dG/dx_i = tr[ (dG/dX).t * dX/dx_ij ] = (W_?i).t * dG/dX.
         *
         * Thus dG/dx = W.t * dG/dX.
         *
         * Chain Rule (Quadratic weight part) : dG/dx_ij = tr[ ( dG/dX ).t * dX/dx_ij ].
         *
         * Note that x is a column vector with inA, inB as parts.
         * Because X = inA.t * Q * inB, dX/dxA = inB.t * Q.t and dX/dxB = inA.t * Q
         * Since dG/dX is scalar, we obtain dG/dx by scalar multiplication.
         */
        val dXdxQ1: DataVec = quadratic(id).value * inB //d tr(ax')/dx = d tr(x'a)/dx = a'
      val dXdxQ2: DataVec = quadratic(id).value.t * inA //d tr(ax)/dx = d tr(xa)/dx = a
        dGdx += restoreError(dXdxQ1, dXdxQ2) :* dGdXi

        dGdQ
      }

      ((dGdX, dGdL, dGdQ), dGdx)
    }.unzip

    val (dX, dL, dQ) = internal.unzip3

    bias update dX
    linear update dL
    (0 until NOut).par.foreach { id ⇒
      quadratic(id) update dQ(id)
    }

    external.seq
  }

  override def initiateBy(builder: WeightBuilder): this.type = {
    if (NIn > 0 && NOut > 0) {
      val range = act.initialize(NIn, NOut)
      if (quadratic.isEmpty)
        (0 until NOut).foreach { _ ⇒
          quadratic += new Weight[Matrix]
        }
      quadratic.foreach(weight ⇒ builder.buildMatrix(weight, fanInA, fanInB, range))
      builder.buildMatrix(linear, NOut, NIn, range)
      builder.buildVector(bias, NOut, range)
    }

    this
  }

  override def loss: Double = linear.loss + bias.loss + quadratic.map(_.loss).sum

  override def read(kryo: Kryo, input: Input): Unit = {
    fanInA = input.readInt()
    fanInB = input.readInt()
    act = kryo.readClassAndObject(input).asInstanceOf[Activation]
    bias.read(kryo, input)
    linear.read(kryo, input)
    val size = input.readInt()
    (0 until size).map { _ ⇒
      val weight = new Weight[Matrix]
      weight.read(kryo, input)
      quadratic += weight
    }
    super.read(kryo, input)
  }

  override def write(kryo: Kryo, output: Output): Unit = {
    output.writeInt(fanInA)
    output.writeInt(fanInB)
    kryo.writeClassAndObject(output, act)
    bias.write(kryo, output)
    linear.write(kryo, output)
    output.writeInt(NOut)
    quadratic.foreach(_.write(kryo, output))
    super.write(kryo, output)
  }
}