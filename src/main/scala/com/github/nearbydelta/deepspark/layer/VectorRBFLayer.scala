package com.github.nearbydelta.deepspark.layer

import breeze.linalg._
import breeze.numerics.{abs, pow}
import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.nearbydelta.deepspark.data.{Matrix, _}

import scala.collection.parallel.ParSeq

/**
 * __Layer__ : An Radial Basis Function Layer, with its radial basis.
 *
 * @note This is a RBF layer, mainly the same with 3-phrase RBF in paper
 *       [[http://www.sciencedirect.com/science/article/pii/S0893608001000272 Three learning phrases for radial-basis-function networks]]
 */
class VectorRBFLayer extends TransformLayer {
  /** Centroids **/
  val centers: Weight[Matrix] = new Weight[Matrix]
  /** Epsilon values vector **/
  val epsilon: Weight[Matrix] = new Weight[Matrix]
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
   * @param minDistances Minimum number of distances will be used in epsilon initialization.
   * @param alpha Scaling paramater alpha, when determining epsilon.
   * @return self
   */
  def withCenters(centroid: Seq[DataVec], minDistances: Int = 3, alpha: Double = 1.0): this.type = {
    NIn = centroid.head.size
    NOut = centroid.size

    val zipped = centroid.zipWithIndex
    val matrix = DenseMatrix.zeros[Double](NIn, NOut)
    zipped.foreach {
      case (c, id) ⇒ matrix(::, id) := c
    }
    centers.withValue(matrix)

    val eps = DenseMatrix.fill[Double](NIn, NOut)(1 / Math.sqrt(2))
    zipped.combinations(2).flatMap { seq ⇒
      val (c1, id1) = seq.head
      val (c2, id2) = seq(1)
      val d = abs(c1 - c2)
      Seq((id1, d), (id2, d))
    }.toSeq.groupBy(_._1).foreach {
      case (id, seq) ⇒
        (0 until NIn).foreach { r ⇒
          val d = seq.map(_._2(r)).sorted.take(minDistances).sum / minDistances
          // Vector form has diag(e_1, e_2, ...) = diag(1/\sqrt{2}s_1, 1/\sqrt{2}s_2, ...)
          // This is because we use form (v-c)^T * E * E * (v-c).
          // The paper suggests, s_j = alpha * average distance
          val sigma = alpha * d
          // Hence, e_j = 1/(sqrt(2)*alpha*average distance).
          eps(r, id) :/= sigma
        }
    }
    epsilon.withValue(eps)

    this
  }

  override def apply(x: DataVec): DataVec =
    if (x != null) {
      val diff = centers.value(::, *) - x
      val dist = pow(epsilon.value :* diff, 2.0)
      val wr = sum(dist(::, *)).toDenseVector
      act(wr)
    } else
      DenseVector.zeros[Double](NOut)

  override def backprop(seq: ParSeq[((DataVec, DataVec), DataVec)]): (ParSeq[DataVec], ParSeq[() ⇒ Unit]) = {
    val (dE, dC, external) = seq.collect { case ((in, out), error) if in != null ⇒
      val diff = centers.value(::, *) - in
      val dist = 2.0 * (epsilon.value :* diff)

      /*
     * Error, i.e. dGdF is NOut-dim vector.
     * act.derivative is NOut x NOut matrix.
     * Actually, dFdX is NOut-dim square matrix. Since non-diag is zero,
     * dFdX * dGdF can be implemented with elementwise computation.
     * Thus dGdX = dFdX * dGdF, NOut-dim vector
     */
      val dGdX: DataVec = act.derivative(out) :* error

      /*
       * dX_i/dC_ij = 2 * e_ij^2 :* (x_j - C_ij), otherwise 0 vector.
       * dX/dC_i became all zero except i-th column is dX_i/dC_i.
       * Therefore, while computing dGdC,
       * dG/dC_i = dGdX * dXdC_i = dGdX(i) * dXdC_i.
       */
      val dGdC = epsilon.value :* dist

      /*
       * Note that, dX_i/dC_i = - dX_i/dx, and
       * dX/dx = - \sum dX_i/dC_i.
       */
      val dGdx = DenseVector.zeros[Double](diff.rows)

      /*
       * dX_i/dE_ij = 2 * e_ij * (x_j - C_ij)^2, otherwise 0.
       * Then dGdE = dXdE * dGdX is NOut-dim vector.
       * Simplified version will be, elementwise multiplication between diag(dXdE) and dGdX
       */
      val dGdE = dist :* diff

      (0 until NOut).par.foreach { j ⇒
        val d = dGdX(j)
        dGdC(::, j) *= d
        dGdE(::, j) *= d
      }

      (dGdE, dGdC, dGdx)
    }.unzip3

    (external, ParSeq(
      epsilon -= dE,
      centers -= dC
    ))
  }

  override def initiateBy(builder: WeightBuilder): this.type = {
    if (builder != null && NIn > 0 && NOut > 0) {
      builder.buildMatrix(epsilon, NIn, NOut, noReg = false)
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
