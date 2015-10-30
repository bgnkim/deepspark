package com.github.nearbydelta.deepspark

import breeze.linalg.{DenseMatrix, DenseVector, norm}

/**
 * Created by bydelta on 15. 10. 14.
 */
package object data {
  private final val UpperPattern = """(?U)\p{IsUppercase}+""".r
  private final val LowerPattern = """(?U)\p{IsLowercase}+""".r
  private final val DigitPattern = """(?U)\p{IsDigit}+""".r
  private final val PunctPattern = """(?U)\p{IsPunctuation}+""".r

  type DataVec = breeze.linalg.DenseVector[Double]
  type Matrix = breeze.linalg.DenseMatrix[Double]

  trait Differentiable extends Serializable with (DataVec ⇒ DataVec) {
    def diffAtY(y: DataVec): Matrix = derivative(y)

    def derivative(fx: DataVec): Matrix
  }

  trait Zero[X] extends (X ⇒ X)

  implicit object WeightZero extends Zero[Matrix] {
    override def apply(w: Matrix): Matrix =
      DenseMatrix.zeros[Double](w.rows, w.cols)
  }

  implicit object VectorZero extends Zero[DataVec] {
    override def apply(w: DataVec): DataVec =
      DenseVector.zeros[Double](w.length)
  }

  implicit object MatrixNorm extends norm.Impl[Matrix, Double] {
    override def apply(v: Matrix): Double = v.valuesIterator.map(x ⇒ x * x).sum
  }

  implicit class StringShapeOps(x: String) {
    def getShape = {
      val shape = PunctPattern.replaceAllIn(
        DigitPattern.replaceAllIn(
          LowerPattern.replaceAllIn(
            UpperPattern.replaceAllIn(x, "A"), "a"
          ), "0"
        ), "."
      ).replaceAll("[^Aa0\\.]+", "x")
      "#SHAPE_" + shape
    }
  }

  val HardInverseQuadRBF = HardGaussianRBF
}
