package com.github.nearbydelta.deepspark

import breeze.linalg.{DenseMatrix, DenseVector, norm}
import com.github.fommil.netlib.BLAS.{getInstance ⇒ blas}

/**
 * Package: deepspark.data
 *
 * This package contains basically required classes and objects.
 */
package object data {
  /** Type: Vector **/
  type DataVec = breeze.linalg.DenseVector[Double]
  /** Type: Matrix **/
  type Matrix = breeze.linalg.DenseMatrix[Double]
  /** Numeric Letter pattern **/
  private final val DigitPattern = """(?U)\p{IsDigit}+""".r
  /** Lowercase Letter pattern **/
  private final val LowerPattern = """(?U)\p{IsLowercase}+""".r
  /** Punctuation Letter pattern **/
  private final val PunctPattern = """(?U)\p{IsPunctuation}+""".r
  /** Uppercase Letter pattern **/
  private final val UpperPattern = """(?U)\p{IsUppercase}+""".r
  /** Alias for Hard version of Inverse Quadratic RBF. **/
  val HardInverseQuadRBF = HardGaussianRBF

  /**
   * return aAx + by
   * @param a Double, scalar
   * @param A Matrix
   * @param x DataVec
   * @param b Double, scalar
   * @param y DataVec
   */
  def aAxpby(a: Double, A: Matrix, x: DataVec, b: Double, y:DataVec) = {
    val rv = y.copy

    require(x.length == A.cols, s"Dimension mismatch! Matrix x ${A.cols} columns != Vector a ${x.length} rows.")
    require(A.rows == y.length, s"Dimension mismatch! Matrix x ${A.rows} rows != Vector a ${y.length} rows.")

    blas.dgemv(if(A.isTranspose) "T" else "N",
      if(A.isTranspose) A.cols else A.rows, if(A.isTranspose) A.rows else A.cols,
      a, A.data, A.offset, A.majorStride,
      x.data, x.offset, x.stride,
      b, rv.data, rv.offset, rv.stride)
    rv
  }

  def matrixAxpy(A: Matrix, x: DataVec, y: DataVec) = aAxpby(1.0, A, x, 1.0, y)

  /**
   * Trait of Differentiable function over Vector.
   */
  trait Differentiable extends Serializable with (DataVec ⇒ DataVec) {
    /**
     * Differentiate function at output value fx.
     * @param fx Differentiation point vector, by output value.
     * @return Differentiation value of this function, generally squared matrix with same dimension.
     */
    def derivative(fx: DataVec): Matrix

    /**
     * Differentiate function at output value y.
     * @param y Differentiation point vector, by output value.
     * @return Differentiation value of this function, generally squared matrix with same dimension.
     */
    def diffAtY(y: DataVec): Matrix = derivative(y)
  }

  /**
   * Trait that gives zero value of given type
   * @tparam X Type parameter of zero value will be given.
   */
  trait Zero[X] extends (X ⇒ X)

  /**
   * Implicit class for getting shape of given string.
   * @param x String which shape will be extracted.
   */
  implicit class StringShapeOps(x: String) {
    /**
     * Get shape of string
     *
    - Prefix: `#SHAPE_` to distinguish from normal words.
     - Uppercase: sequence will be substituted by single letter `A`
     - Lowercase: sequence will be substituted by single letter `a`
     - Digit: sequence will be substituted by single letter `0`
     - Punctuation: sequence will be substituted by single letter `.`
     - Others: sequence will be substituted by single letter `x`
     *
     * All the European upper/lower cases will be substituted by `A/a`.
     *
     * @return Shape string.
     */
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

  /**
   * Implicit object for Zero value of matrix
   */
  implicit object WeightZero extends Zero[Matrix] {
    override def apply(w: Matrix): Matrix =
      DenseMatrix.zeros[Double](w.rows, w.cols)
  }

  /**
   * Implicit object for Zero value of vector
   */
  implicit object VectorZero extends Zero[DataVec] {
    override def apply(w: DataVec): DataVec =
      DenseVector.zeros[Double](w.length)
  }

  /**
   * Implicit object for Frobenious norm of matrix
   */
  implicit object MatrixNorm extends norm.Impl[Matrix, Double] {
    override def apply(v: Matrix): Double = v.valuesIterator.map(x ⇒ x * x).sum
  }
}
