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
   * Initialization range of weight
   *
   * @param fanIn the number of __fan-in__ ''i.e. the number of neurons in previous layer''
   * @param fanOut the number of __fan-out__ ''i.e. the number of neurons in next layer''
   * @return the initialized range of weight
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
 *            val diff = HardSigmoid.derivative(fx) }}}
 */
object HardSigmoid extends Activation {
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

  override def derivative(fx: DataVec): Matrix = {
    // Because fx is n by 1 matrix, generate n by n matrix
    val res = DenseMatrix.zeros[Double](fx.length, fx.length)
    // Output is diagonal matrix, with dfi(xi)/dxi.
    derivCoord(fx, res, fx.length - 1)
  }
}

/**
 * __Activation Function__: Hard version of Tanh (Hyperbolic Tangent)
 *
 * @note `tanh(x) = sinh(x) / cosh(x)`, hard version approximates tanh as piecewise linear function.
 *       We assumed the input of activation is a row vector.
 * @example
 * {{{val fx = HardTanh(0.0)
 *            val diff = HardTanh.derivative(fx) }}}
 */
object HardTanh extends Activation {
  @tailrec
  private def applyCoord(x: DataVec, res: DataVec, r: Int): DataVec =
    if (r >= 0) {
      val v = x(r)
      if (v < -1) res.update(r, -1.0)
      else if (v > 1) res.update(r, 1.0)

      applyCoord(x, res, r - 1)
    } else
      res

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

  override def derivative(fx: DataVec): Matrix = {
    // Because fx is n by 1 matrix, generate n by n matrix
    val res = DenseMatrix.zeros[Double](fx.length, fx.length)
    // Output is diagonal matrix, with dfi(xi)/dxi.
    derivCoord(fx, res, fx.length - 1)
  }
}

/**
 * __Activation Function__: Tanh (Hyperbolic Tangent)
 *
 * @note `tanh(x) = sinh(x) / cosh(x)`
 *       We assumed the input of activation is a row vector.
 * @example
 * {{{val fx = HyperbolicTangent(0.0)
 *           val diff = HyperbolicTangent.derivative(fx) }}}
 */
object HyperbolicTangent extends Activation {
  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  override def apply(x: DataVec): DataVec = tanh(x)

  override def derivative(fx: DataVec): Matrix = {
    // Output is diagonal matrix, with dfi(xi)/dxi.
    val dVec: DataVec = 1.0 - (fx :* fx)
    diag(dVec)
  }
}

/**
 * __Activation Function__: Linear
 *
 * @note `linear(x) = x`
 *       We assumed the input of activation is a row vector.
 * @example
  * {{{val fx = Linear(0.0)
 *                       val diff = Linear.derivative(fx)}}}
 */
object Linear extends Activation {
  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  override def apply(x: DataVec): DataVec = x.copy

  override def derivative(fx: DataVec): Matrix = DenseMatrix.eye[Double](fx.length)
}

/**
 * __Activation Function__: Rectifier
 *
 * @note `rectifier(x) = x if x > 0, otherwise 0`
 *       We assumed the input of activation is a row vector.
 * @example
 * {{{val fx = Rectifier(0.0)
 *           val diff = Rectifier.derivative(fx)}}}
 */
object Rectifier extends Activation {
  @tailrec
  private def applyCoord(x: DataVec, res: DataVec, r: Int): DataVec =
    if (r >= 0) {
      if (x(r) < 0) res.update(r, 0.0)
      applyCoord(x, res, r - 1)
    } else
      res

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

  override def derivative(fx: DataVec): Matrix = {
    // Because fx is n by 1 matrix, generate n by n matrix
    val res = DenseMatrix.zeros[Double](fx.length, fx.length)
    // Output is diagonal matrix, with dfi(xi)/dxi.
    derivCoord(fx, res, fx.length - 1)
  }
}


/**
 * __Activation Function__: LeakyReLU
 *
 * @note `rectifier(x) = x if x > 0, otherwise 0.01x`
 *       We assumed the input of activation is a row vector.
 * @example
 * {{{val fx = LeakyReLU(0.0)
 *           val diff = LeakyReLU.derivative(fx)}}}
 */
object LeakyReLU extends Activation {
  @tailrec
  private def applyCoord(x: DataVec, res: DataVec, r: Int): DataVec =
    if (r >= 0) {
      if (x(r) < 0) res(r) *= 0.01
      applyCoord(x, res, r - 1)
    } else
      res

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

  override def derivative(fx: DataVec): Matrix = {
    // Because fx is n by 1 matrix, generate n by n matrix
    val res = DenseMatrix.ones[Double](fx.length, fx.length)
    // Output is diagonal matrix, with dfi(xi)/dxi.
    derivCoord(fx, res, fx.length - 1)
  }
}

/**
 * __Activation Function__: Sigmoid function
 *
 * @note {{{sigmoid(x) = 1 / [exp(-x) + 1]}}}
 *       We assumed the input of activation is a row vector.
 * @example
 * {{{val fx = Sigmoid(0.0)
 *           val diff = Sigmoid.derivative(fx)}}}
 */
object Sigmoid extends Activation {
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

  override def derivative(fx: DataVec): Matrix = {
    // Output is diagonal matrix, with dfi(xi)/dxi.
    val dVec: DataVec = (1.0 - fx) :* fx
    diag(dVec)
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
 *            val diff = Softmax.derivative(fx)}}}
 */
object Softmax extends Activation {
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

  override def derivative(fx: DataVec): Matrix = {
    val res: Matrix = DenseMatrix.eye[Double](fx.length)
    res(::, *) -= fx
    fx.foreachPair {
      case (k, v) ⇒
        res(k, ::) *= v
    }

    // Note that (i, j)-entry of deriviative is dF_j / dX_i
    // and dF_j / dX_i = F(j) * (Delta_ij - F(i)).
    res
  }

  override def initialize(fanIn: Int, fanOut: Int): (Double, Double) = {
    val range = (Math.sqrt(6.0 / (fanIn + fanOut)) * 4.0).toFloat
    (-range, range)
  }
}

/**
 * __Activation Function__: Softmax function for [[CrossEntropyErr]]
 *
 * @note {{{softmax(x)_i = exp(x_i) / sum(exp(x_i))}}}
 *       We assumed the input of activation is a row vector.<br/>
 *       We assumed the error function is [[CrossEntropyErr]].
 * @example
 * {{{val fx = Softmax(0.0)
 *              val diff = Softmax.derivative(fx)}}}
 */
object SoftmaxCEE extends Activation {
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
   * @inheritdoc
   * @note Assumes [[CrossEntropyErr]] objective function, and in that case,
   *       CEE's derivative -1/out will be canceled with Softmax derivation term out(delta_ij-out)in.
   *       Hence, we designed softmax with derivation term (delta_ij-out)in.
   */
  override def derivative(fx: DataVec): Matrix = {
    // Note that (i, j)-entry of deriviative is dF_j / dX_i
    // and dF_j / dX_i = F(j) * (Delta_ij - F(i)).
    // but since F(j) will be canceled, we used only Delta_ij - F(i).
    // This can be easily implemented by copying -F for every column,
    // And add 1 for diagonal entries.

    val res: Matrix = DenseMatrix.eye[Double](fx.length)
    res(::, *) -= fx
    res
  }

  override def initialize(fanIn: Int, fanOut: Int): (Double, Double) = {
    val range = (Math.sqrt(6.0 / (fanIn + fanOut)) * 4.0).toFloat
    (-range, range)
  }
}

/**
 * __Activation Function__: Softplus
 *
 * @note `softplus(x) = log[1 + exp(x)]`
 *       We assumed the input of activation is a row vector.
 * @example
 * {{{val fx = Softplus(0.0)
 *           val diff = Softplus.derivative(fx)}}}
 */
object Softplus extends Activation {
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

  override def derivative(fx: DataVec): Matrix = {
    // Output is diagonal matrix, with dfi(xi)/dxi.
    val expv: DataVec = expm1(fx)
    val exp1: DataVec = expv - 1.0
    val dVec: DataVec = exp1 / expv
    diag(dVec)
  }
}