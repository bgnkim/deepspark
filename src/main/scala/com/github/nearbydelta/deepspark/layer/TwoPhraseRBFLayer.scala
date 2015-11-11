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
class TwoPhraseRBFLayer extends TransformLayer {
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
   * @param minDistances Minimum number of distances will be used in epsilon initialization.
   * @param alpha Scaling paramater alpha, when determining epsilon.
   * @return self
   */
  def withCenters(centroid: Seq[DataVec], minDistances: Int = 3, alpha: Double = 1.0) = {
    NIn = centroid.head.size
    NOut = centroid.size

    val zipped = centroid.zipWithIndex
    val matrix = DenseMatrix.zeros[Double](NIn, NOut)
    zipped.foreach {
      case (c, id) ⇒ matrix(::, id) := c
    }
    centers.withValue(matrix)

    val eps = DenseVector.zeros[Double](NOut)
    zipped.combinations(2).flatMap { seq ⇒
      val (c1, id1) = seq.head
      val (c2, id2) = seq(1)
      val d = norm(c1 - c2)
      Seq((id1, d), (id2, d))
    }.toSeq.groupBy(_._1).foreach {
      case (id, seq) ⇒
        val d = seq.map(_._2).sorted.take(minDistances).sum / minDistances
        // Since the paper assumes form of d^2 / 2s_j^2, but we're using (e_j*d)^2.
        // Note that e_j = 1/(sqrt(2)*s_j). The paper suggests, s_j = alpha * average distance
        val sigma = alpha * d
        val eInv = Math.sqrt(2) * sigma
        // Hence, e_j = 1/(sqrt(2)*alpha*average distance).
        eps.update(id, 1 / eInv)
    }
    epsilon.withValue(eps)

    this
  }

  override def apply(x: DataVec): DataVec =
    if (x != null) {
      val diff = centers.value(::, *) - x
      val distances = norm.apply(diff, Axis._0).toDenseVector
      val wr = epsilon.value :* distances
      act(pow(wr, 2.0))
    } else
      DenseVector.zeros[Double](NOut)

  override def backprop(seq: ParSeq[((DataVec, DataVec), DataVec)]): ParSeq[DataVec] = {
    val (dE, external) = seq.collect { case ((in, out), error) if in != null ⇒
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
       * Note that, dX_i/dC_i = - dX_i/dx, and
       * dX/dx = - \sum dX_i/dC_i.
       */
      val dGdx = DenseVector.zeros[Double](diff.rows)

      /*
       * dX_i/dE_i = 2 * e_i * ||x - C_i||^2, otherwise 0.
       * Then dGdE = dXdE * dGdX is NOut-dim vector.
       * Simplified version will be, elementwise multiplication between diag(dXdE) and dGdX
       */
      val dGdE = dist.mapPairs {
        case (i, d) ⇒
          val e = epsilon.value(i)
          val common = 2.0 * e * dGdX(i)
          val vec: DataVec = diff(::, i) :* (common * e)
          dGdx += vec
          common * d * d
      }

      (dGdE, dGdx)
    }.unzip

    epsilon update dE

    external
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
