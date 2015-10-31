package com.github.nearbydelta.deepspark.layer

import breeze.linalg._
import breeze.numerics.pow
import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.nearbydelta.deepspark.data.{Matrix, _}

import scala.collection.parallel.ParSeq

/**
 * __Layer__ : An Radial Basis Function Layer, with its radial basis.
 */
class RBFLayer extends TransformLayer {
  val epsilon: Weight[DataVec] = new Weight[DataVec]
  val centers: Weight[Matrix] = new Weight[Matrix]
  var act: RBFActivation = GaussianRBF

  def withActivation(act: RBFActivation): this.type = {
    this.act = act
    this
  }

  override def initiateBy(builder: WeightBuilder): this.type = {
    if (builder != null && NIn > 0 && NOut > 0) {
      builder.buildVector(epsilon, NOut)
      builder.buildMatrix(centers, NIn, NOut, noReg = true)
    }

    this
  }

  @deprecated
  override def withInput(in: Int): this.type = this

  @deprecated
  override def withOutput(out: Int): this.type = this

  def withCenters(centroid: Seq[DataVec]) = {
    NIn = centroid.head.size
    NOut = centroid.size

    val matrix = DenseMatrix.zeros[Double](NIn, NOut)
    centroid.zipWithIndex.foreach {
      case (c, id) ⇒ matrix(::, id) := c
    }

    centers.withValue(matrix)(WeightZero)
    this
  }

  override def write(kryo: Kryo, output: Output): Unit = {
    kryo.writeClassAndObject(output, act)
    epsilon.write(kryo, output)
    centers.write(kryo, output)
    super.write(kryo, output)
  }

  override def read(kryo: Kryo, input: Input): Unit = {
    act = kryo.readClassAndObject(input).asInstanceOf[RBFActivation]
    epsilon.read(kryo, input)
    centers.read(kryo, input)
    super.read(kryo, input)
  }

  override def loss: Double = epsilon.loss

  override def update(count: Int): Unit = {
    epsilon.update(count)
    centers.update(count)
  }

  /**
   * Forward computation
   *
   * @param x input matrix
   * @return output matrix
   */
  override def apply(x: DataVec): DataVec = {
    val diff = centers.value(::, *) - x
    val distances = norm.apply(diff, Axis._0).toDenseVector
    val wr = epsilon.value :* distances
    act(pow(wr, 2.0))
  }

  /**
   * <p>Backward computation.</p>
   *
   * @note <p>
   *       Let this layer have function F composed with function <code> X_i(x) = (e_i :* ||x - C_i||) ** 2 </code>
   *       and higher layer have function G.
   *       </p>
   *
   *       <p>
   *       Epsilon is updated with: <code>dG/dE = dG/dF dF/dX dX/dE</code>
   *       Center is updated with: <code>dG/dC_i = dG/dF dF/dX dX/dC_i</code>
   *       and propagate <code>dG/dx</code> which is the same as <code>\sum dG/dC_i </code>
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
    val (dE, dC, external) = seq.map { case ((in, out), error) ⇒
      val diff = centers.value(::, *) - in
      val dist = norm.apply(diff, Axis._0).toDenseVector
      /*
     * Error, i.e. dGdF is NOut-dim vector.
     * act.derivative is NOut x NOut matrix.
     * Actually, dFdX is NOut-dim square matrix. Since non-diag is zero,
     * dFdX * dGdF can be implemented with elementwise computation.
     * Thus dGdX = dFdX * dGdF, NOut-dim vector
     */
      val dGdX: DataVec = act.derivative(out) :* error

      /*
     * dX_i/dE_i = 2 * e_i * ||x - C_i||^2, otherwise 0.
     * Then dGdE = dXdE * dGdX is NOut-dim vector.
     * Simplified version will be, elementwise multiplication between diag(dXdE) and dGdX
     */
      val dXdE = 2.0 * (epsilon.value :* pow(dist, 2.0))
      val dGdE = dXdE :* dGdX

      /*
     * dX_i/dC_i = 2 * e_i^2 * (x - C_i), otherwise 0 vector.
     * dX/dC_i became all zero except i-th column is dX_i/dC_i.
     * Therefore, while computing dGdC,
     * dG/dC_i = dGdX * dXdC_i = dGdX(i) * dXdC_i.
     */
      val factor = 2.0 * (pow(epsilon.value, 2.0) :* dGdX)
      val dGdC = diff(*, ::) :* factor

      /*
     * Note that, dX_i/dC_i = - dX_i/dx, and
     * dX/dx = - \sum dX_i/dC_i.
     */
      val dGdx = sum(dGdC, Axis._1)
      (dGdE, dGdC, dGdx)
    }.unzip3

    epsilon updateBy dE.reduce(_ += _)
    centers updateBy dC.reduce(_ += _)

    external.seq
  }
}
