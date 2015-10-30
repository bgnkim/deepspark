package com.github.nearbydelta.deepspark.data

import breeze.linalg._
import breeze.numerics._

import scala.annotation.tailrec

/**
 * __Trait__ that describes an activation function for '''each layer'''
 *
 * Because these activation functions can be shared, we recommend to make inherited one as an object.
 */
trait Activation extends Differentiable {
  /**
   * Initialize the weight matrix
   *
   * @param fanIn the number of __fan-in__ ''i.e. the number of neurons in previous layer''
   * @param fanOut the number of __fan-out__ ''i.e. the number of neurons in next layer''
   * @return the initialized weight matrix
   */
  def initialize(fanIn: Int, fanOut: Int): (Double, Double) = {
    val range = Math.sqrt(6.0 / (fanIn + fanOut))
    (-range, range)
  }
}

/**
 * __Activation Function__: Hard version of Sigmoid
 *
 * @note `sigmoid(x) = 1 / [exp(-x) + 1]`, hard version approximates tanh as piecewise linear function
 *       (derived from relationship between tanh & sigmoid, and tanh & hard tanh.)
 *       We assumed the input of activation is a row vector.
 * @example
 * {{{val fx = HardSigmoid(0.0)
 *           val diff = HardSigmoid.derivative(fx) }}}
 */
object HardSigmoid extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, symmetric matrix__
   */
  override def derivative(fx: DataVec): Matrix = {
    // Because fx is n by 1 matrix, generate n by n matrix
    val res = DenseMatrix.zeros[Double](fx.length, fx.length)
    // Output is diagonal matrix, with dfi(xi)/dxi.
    derivCoord(fx, res, fx.length - 1)
  }

  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  override def apply(x: DataVec): DataVec = {
    val res = DenseVector.zeros[Double](x.length)
    applyCoord(x, res, x.length - 1)
  }

  @tailrec
  private def derivCoord(fx: DataVec, res: Matrix, r: Int): Matrix =
    if (r >= 0) {
      val x = fx(r)
      if (x > 0.0 && x < 1.0)
        res.update(r, r, 0.25)
      // else res.update((r, r), 0.0f) [Already initialized as zero]
      derivCoord(fx, res, r - 1)
    } else
      res

  @tailrec
  private def applyCoord(x: DataVec, res: DataVec, r: Int): DataVec =
    if (r >= 0) {
      val v = x(r)
      // if (v < -2) res.update(r, c, 0.0f) [Already initailized as zero]
      if (v > 2) res.update(r, 1.0)
      else res.update(r, 0.25 * v + 0.5)

      applyCoord(x, res, r - 1)
    } else
      res
}

/**
 * __Activation Function__: Hard version of Tanh (Hyperbolic Tangent)
 *
 * @note `tanh(x) = sinh(x) / cosh(x)`, hard version approximates tanh as piecewise linear function.
 *       We assumed the input of activation is a row vector.
 * @example
 * {{{val fx = HardTanh(0.0)
 *           val diff = HardTanh.derivative(fx) }}}
 */
object HardTanh extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, symmetric matrix__
   */
  override def derivative(fx: DataVec): Matrix = {
    // Because fx is n by 1 matrix, generate n by n matrix
    val res = DenseMatrix.zeros[Double](fx.length, fx.length)
    // Output is diagonal matrix, with dfi(xi)/dxi.
    derivCoord(fx, res, fx.length - 1)
  }

  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  override def apply(x: DataVec): DataVec = {
    val res = x.copy
    applyCoord(x, res, x.length - 1)
  }

  @tailrec
  private def derivCoord(fx: DataVec, res: Matrix, r: Int): Matrix =
    if (r >= 0) {
      val x = fx(r)
      if (x < 1.0 && x > -1.0)
        res.update(r, r, 1.0)
      // else res.update(r, r, 0.0f) [Already initalized as zero]
      derivCoord(fx, res, r - 1)
    } else
      res

  @tailrec
  private def applyCoord(x: DataVec, res: DataVec, r: Int): DataVec =
    if (r >= 0) {
      val v = x(r)
      if (v < -1) res.update(r, -1.0)
      else if (v > 1) res.update(r, 1.0)

      applyCoord(x, res, r - 1)
    } else
      res
}

/**
 * __Activation Function__: Tanh (Hyperbolic Tangent)
 *
 * @note `tanh(x) = sinh(x) / cosh(x)`
 *       We assumed the input of activation is a row vector.
 * @example
 * {{{val fx = HyperbolicTangent(0.0)
 *          val diff = HyperbolicTangent.derivative(fx) }}}
 */
object HyperbolicTangent extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, symmetric matrix__
   */
  override def derivative(fx: DataVec): Matrix = {
    // Output is diagonal matrix, with dfi(xi)/dxi.
    val dVec: DataVec = 1.0 - (fx :* fx)
    diag(dVec)
  }

  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  override def apply(x: DataVec): DataVec = tanh(x)
}

/**
 * __Activation Function__: Linear
 *
 * @note `linear(x) = x`
 *       We assumed the input of activation is a row vector.
 * @example
  * {{{val fx = Linear(0.0)
 *                     val diff = Linear.derivative(fx)}}}
 */
object Linear extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, symmetric matrix__
   */
  override def derivative(fx: DataVec): Matrix = DenseMatrix.eye[Double](fx.length)

  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  override def apply(x: DataVec): DataVec = x.copy
}

/**
 * __Activation Function__: Rectifier
 *
 * @note `rectifier(x) = x if x > 0, otherwise 0`
 *       We assumed the input of activation is a row vector.
 * @example
 * {{{val fx = Rectifier(0.0)
 *          val diff = Rectifier.derivative(fx)}}}
 */
object Rectifier extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, symmetric matrix__
   */
  override def derivative(fx: DataVec): Matrix = {
    // Because fx is n by 1 matrix, generate n by n matrix
    val res = DenseMatrix.zeros[Double](fx.length, fx.length)
    // Output is diagonal matrix, with dfi(xi)/dxi.
    derivCoord(fx, res, fx.length - 1)
  }

  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  override def apply(x: DataVec): DataVec = {
    val res = x.copy
    applyCoord(x, res, x.length - 1)
  }

  @tailrec
  private def derivCoord(fx: DataVec, res: Matrix, r: Int): Matrix =
    if (r >= 0) {
      val x = fx(r)
      if (x > 0)
        res.update(r, r, 1.0)
      //else res.update(r, r, 0.0f) [Already Initialized as zero]
      derivCoord(fx, res, r - 1)
    } else
      res

  @tailrec
  private def applyCoord(x: DataVec, res: DataVec, r: Int): DataVec =
    if (r >= 0) {
      if (x(r) < 0) res.update(r, 0.0)
      applyCoord(x, res, r - 1)
    } else
      res
}


/**
 * __Activation Function__: LeakyReLU
 *
 * @note `rectifier(x) = x if x > 0, otherwise 0.01x`
 *       We assumed the input of activation is a row vector.
 * @example
 * {{{val fx = Rectifier(0.0)
 *          val diff = Rectifier.derivative(fx)}}}
 */
object LeakyReLU extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, symmetric matrix__
   */
  override def derivative(fx: DataVec): Matrix = {
    // Because fx is n by 1 matrix, generate n by n matrix
    val res = DenseMatrix.ones[Double](fx.length, fx.length)
    // Output is diagonal matrix, with dfi(xi)/dxi.
    derivCoord(fx, res, fx.length - 1)
  }

  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  override def apply(x: DataVec): DataVec = {
    val res = x.copy
    applyCoord(x, res, x.length - 1)
  }

  @tailrec
  private def derivCoord(fx: DataVec, res: Matrix, r: Int): Matrix =
    if (r >= 0) {
      val x = fx(r)
      if (x < 0)
        res.update(r, r, 0.01)
      //else res.update(r, r, 0.0f) [Already Initialized as zero]
      derivCoord(fx, res, r - 1)
    } else
      res

  @tailrec
  private def applyCoord(x: DataVec, res: DataVec, r: Int): DataVec =
    if (r >= 0) {
      if (x(r) < 0) res(r) *= 0.01
      applyCoord(x, res, r - 1)
    } else
      res
}

/**
 * __Activation Function__: Sigmoid function
 *
 * @note {{{sigmoid(x) = 1 / [exp(-x) + 1]}}}
 *       We assumed the input of activation is a row vector.
 * @example
 * {{{val fx = Sigmoid(0.0)
 *          val diff = Sigmoid.derivative(fx)}}}
 */
object Sigmoid extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, symmetric matrix__
   */
  override def derivative(fx: DataVec): Matrix = {
    // Output is diagonal matrix, with dfi(xi)/dxi.
    val dVec: DataVec = (1.0 - fx) :* fx
    diag(dVec)
  }

  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  override def apply(x: DataVec): DataVec = {
    val expv: DataVec = exp(-x)
    val exp1: DataVec = expv :+ 1.0
    1.0 / exp1
  }

  /**
   * Initialize the weight matrix
   *
   * @param fanIn the number of __fan-in__ ''i.e. the number of neurons in previous layer''
   * @param fanOut the number of __fan-out__ ''i.e. the number of neurons in next layer''
   * @return the initialized weight matrix
   */
  override def initialize(fanIn: Int, fanOut: Int): (Double, Double) = {
    val range = (Math.sqrt(6.0 / (fanIn + fanOut)) * 4.0).toFloat
    (-range, range)
  }
}

/**
 * __Activation Function__: Softmax function
 *
 * @note {{{softmax(x)_i = exp(x_i) / sum(exp(x_i))}}}
 *       We assumed the input of activation is a row vector.
 * @example
 * {{{val fx = Softmax(0.0)
 *           val diff = Softmax.derivative(fx)}}}
 */
object Softmax extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, symmetric matrix__
   */
  override def derivative(fx: DataVec): Matrix = {
    val res: Matrix = fx * DenseVector.ones[Double](fx.length).t

    // Note that (i, j)-entry of deriviative is dF_i / dX_j
    // and dF_i / dX_j = F(i) * (Delta_ij - F(j)).
    derivCoord(fx, res, res.rows - 1, res.cols - 1)
  }

  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  override def apply(x: DataVec): DataVec = {
    // exp(x_k) / sum(exp(x_i)) is the same for exp(x_k - max(x)) / sum(exp(x_i - max(x)))
    val maxV: Double = max(x)
    val expv: DataVec = exp(x :- maxV)
    val normalize: Double = sum(expv)
    expv :/= normalize
  }

  /**
   * Initialize the weight matrix
   *
   * @param fanIn the number of __fan-in__ ''i.e. the number of neurons in previous layer''
   * @param fanOut the number of __fan-out__ ''i.e. the number of neurons in next layer''
   * @return the initialized weight matrix
   */
  override def initialize(fanIn: Int, fanOut: Int): (Double, Double) = {
    val range = (Math.sqrt(6.0 / (fanIn + fanOut)) * 4.0).toFloat
    (-range, range)
  }

  @tailrec
  private def derivCoord(fx: DataVec, res: Matrix, r: Int, c: Int): Matrix =
    if (r >= 0) {
      val dfdx = (if (r == c) 1 else 0) - fx(c)
      res.update(r, c, res(r, c) * dfdx)

      if (c > 0) derivCoord(fx, res, r, c - 1)
      else derivCoord(fx, res, r - 1, fx.length - 1)
    } else
      res
}

/**
 * __Activation Function__: Softplus
 *
 * @note `softplus(x) = log[1 + exp(x)]`
 *       We assumed the input of activation is a row vector.
 * @example
 * {{{val fx = Softplus(0.0)
 *          val diff = Softplus.derivative(fx)}}}
 */
object Softplus extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, symmetric matrix__
   */
  override def derivative(fx: DataVec): Matrix = {
    // Output is diagonal matrix, with dfi(xi)/dxi.
    val expv: DataVec = expm1(fx)
    val exp1: DataVec = expv - 1.0
    val dVec: DataVec = exp1 / expv
    diag(dVec)
  }

  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  override def apply(x: DataVec): DataVec = {
    val expx: DataVec = exp(x)
    log1p(expx)
  }
}