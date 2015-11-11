package com.github.nearbydelta.deepspark.data

import breeze.numerics.{exp, pow}

/**
 * __Trait__ of Activation function on RBF layer.
 */
trait RBFActivation extends Serializable with (DataVec ⇒ DataVec) {
  /**
   * Derivative of activation on output fx.
   * @param fx Derivation point vector as output.
   * @return Derivation value, only for diagonal entries as vector.
   */
  def derivative(fx: DataVec): DataVec
}

/**
 * Gaussian RBF function, exp(-r^2^).
 *
 * @note This function assumes get squared input. i.e. this is exp(-x), and gets r^2^ as input x.
 */
object GaussianRBF extends RBFActivation {
  override def apply(xSq: DataVec): DataVec = exp(-xSq)

  override def derivative(fx: DataVec): DataVec = -fx
}

/**
 * Inverse Quadratic RBF function, 1 / (1 + r^2^)
 *
 * @note This function assumes get squared input. i.e. this is 1 / (1 + x), and gets r^2^ as input x.
 */
object InverseQuadRBF extends RBFActivation {
  override def apply(xSq: DataVec): DataVec = xSq.mapValues(d ⇒ 1.0 / (d + 1.0))

  override def derivative(fx: DataVec): DataVec = -pow(fx, 2.0)
}

/**
 * Hard version of Gaussian RBF function, 1 - r^2^
 *
 * @note This function assumes get squared input. i.e. this is 1 - x, and gets r^2^ as input x.
 */
object HardGaussianRBF extends RBFActivation {
  override def apply(xSq: DataVec): DataVec = xSq.mapValues(x ⇒ if (x > 1.0) 0.0 else 1.0 - x)

  override def derivative(fx: DataVec): DataVec = fx.mapValues(x ⇒ if (x > 0) -1.0 else 0.0)
}