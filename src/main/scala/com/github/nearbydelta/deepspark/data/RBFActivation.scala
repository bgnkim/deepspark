package com.github.nearbydelta.deepspark.data

import breeze.numerics.{exp, pow}

/**
 * Created by bydelta on 15. 10. 22.
 */
trait RBFActivation extends Serializable with (DataVec ⇒ DataVec) {
  def derivative(fx: DataVec): DataVec
}

object GaussianRBF extends RBFActivation {
  override def derivative(fx: DataVec): DataVec = -fx

  override def apply(xSq: DataVec): DataVec = exp(-xSq)
}

object InverseQuadRBF extends RBFActivation {
  override def derivative(fx: DataVec): DataVec = -pow(fx, 2.0)

  override def apply(xSq: DataVec): DataVec = {
    val sq1: DataVec = xSq :+ 1.0
    1.0 / sq1
  }
}

object HardGaussianRBF extends RBFActivation {
  override def derivative(fx: DataVec): DataVec =
    fx.mapValues(x ⇒ if (x > 0) -1.0 else 0.0)

  override def apply(xSq: DataVec): DataVec = {
    xSq.mapValues(x ⇒ if (x > 1.0) 0.0 else 1.0 - x)
  }
}