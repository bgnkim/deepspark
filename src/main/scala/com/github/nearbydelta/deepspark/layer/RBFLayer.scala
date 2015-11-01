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
  /** Centroids **/
  val centers: Weight[Matrix] = new Weight[Matrix]
  /** Epsilon values vector **/
  val epsilon: Weight[DataVec] = new Weight[DataVec]
  /** RBF Activation **/
  var act: RBFActivation = GaussianRBF

  /**
   * Set the activation function.
   * @param act Activation function.
   * @return self
   */
  def withActivation(act: RBFActivation): this.type = {
    this.act = act
    this
  }

  /**
   * Set centroids
   * @param centroid Sequence of Centroids to be set.
   * @return self
   */
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

  override def apply(x: DataVec): DataVec = {
    val diff = centers.value(::, *) - x
    val distances = norm.apply(diff, Axis._0).toDenseVector
    val wr = epsilon.value :* distances
    act(pow(wr, 2.0))
  }

  override def backward(seq: ParSeq[((DataVec, DataVec), DataVec)]): Seq[DataVec] = {
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

  override def initiateBy(builder: WeightBuilder): this.type = {
    if (builder != null && NIn > 0 && NOut > 0) {
      builder.buildVector(epsilon, NOut)
      builder.buildMatrix(centers, NIn, NOut, noReg = true)
    }

    this
  }

  override def loss: Double = epsilon.loss

  override def read(kryo: Kryo, input: Input): Unit = {
    act = kryo.readClassAndObject(input).asInstanceOf[RBFActivation]
    epsilon.read(kryo, input)
    centers.read(kryo, input)
    super.read(kryo, input)
  }

  override def update(count: Int): Unit = {
    epsilon.update(count)
    centers.update(count)
  }

  @deprecated("Input size automatically set when centroid attached")
  override def withInput(in: Int): this.type = this

  @deprecated("Output size automatically set when centroid attached")
  override def withOutput(out: Int): this.type = this

  override def write(kryo: Kryo, output: Output): Unit = {
    kryo.writeClassAndObject(output, act)
    epsilon.write(kryo, output)
    centers.write(kryo, output)
    super.write(kryo, output)
  }
}
