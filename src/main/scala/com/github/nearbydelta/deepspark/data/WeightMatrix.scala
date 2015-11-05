package com.github.nearbydelta.deepspark.data

import breeze.linalg._
import breeze.linalg.operators._
import breeze.numerics._
import com.esotericsoftware.kryo.io.{Input, Output}
import com.esotericsoftware.kryo.{Kryo, KryoSerializable}

/**
 * __Trait__ that describes the algorithm for weight update
 *
 * Because each weight update requires history, we recommend to make inherited one as a class. 
 */
trait WeightBuilder extends Serializable with KryoSerializable {
  /**
   * Build weight update algorithm for given Weight object.
   * @param weight Weight object of Matrix
   * @param row Size of row
   * @param col Size of column
   * @param range Range for random intitialization
   * @param noReg True if this weight does not require regularization.
   * @return Algorithm attached Weight object.
   */
  final def buildMatrix(weight: Weight[Matrix], row: Int, col: Int,
                        range: (Double, Double) = (1e-2, 2e-2),
                        noReg: Boolean = false): Weight[Matrix] = {
    if (weight.isDefined) {
      buildTo(weight, weight.value, noReg)
    } else {
      val (from, to) = range
      val matx = DenseMatrix.rand[Double](row, col)
      matx :*= (to - from)
      matx :+= from
      buildTo(weight, matx, noReg)
    }
  }

  /**
   * Build weight update algorithm for given Weight object.
   * @param weight Weight object of Vector
   * @param row Size of row
   * @param range Range for random intitialization
   * @param noReg True if this weight does not require regularization.
   * @return Algorithm attached Weight object.
   */
  final def buildVector(weight: Weight[DataVec], row: Int,
                        range: (Double, Double) = (1e-2, 2e-2),
                        noReg: Boolean = false): Weight[DataVec] = {
    if (weight.isDefined) {
      buildTo(weight, weight.value, noReg)
    } else {
      val (from, to) = range
      val matx = DenseVector.rand[Double](row)
      matx :*= (to - from)
      matx :+= from
      buildTo(weight, matx, noReg)
    }
  }

  /**
   * Get Update Algorithm for given value.
   * @param value Target of alorithm.
   * @param noReg True if this value does not require regularization.
   * @return Algorithm object
   */
  protected def getUpdater(value: Matrix, noReg: Boolean): Algorithm[Matrix]

  /**
   * Get Update Algorithm for given value.
   * @param value Target of alorithm.
   * @param noReg True if this value does not require regularization.
   * @return Algorithm object
   */
  protected def getUpdater(value: DataVec, noReg: Boolean): Algorithm[DataVec]

  /**
   * Build weight update algorithm for given Weight object.
   * @param weight Weight object of Matrix
   * @param value target Matrix
   * @param noReg True if this weight does not require regularization.
   * @return Algorithm attached Weight object.
   */
  private def buildTo(weight: Weight[Matrix], value: Matrix, noReg: Boolean): Weight[Matrix] = {
    weight withValue value
    weight withUpdater getUpdater(value, noReg)
    weight
  }

  /**
   * Build weight update algorithm for given Weight object.
   * @param weight Weight object of Vector
   * @param value target Vector
   * @param noReg True if this weight does not require regularization.
   * @return Algorithm attached Weight object.
   */
  private def buildTo(weight: Weight[DataVec], value: DataVec, noReg: Boolean): Weight[DataVec] = {
    weight withValue value
    weight withUpdater getUpdater(value, noReg)
    weight
  }
}

/**
 * Companion object for Weight. This class adjusts clipping behavior.
 *
 * For more information, see the paper,
 * [[http://www.jmlr.org/proceedings/papers/v28/pascanu13.pdf On the difficulty of training recurrent neural networks by Pascanu et al., 2013.]]
 */
object Weight {
  /**
   * Threshold for scaling down.
   */
  private var clipping: Option[Double] = None

  /**
   * Check scale, and do scaling down if required.
   * @param x Target value
   * @return Scaled down value.
   */
  def check[K, X <: Tensor[K, Double]](x: X)
                                      (implicit normImpl: norm.Impl[X, Double],
                                       mul: OpMulScalar.Impl2[X, Double, X]): X = {
    clearInvalid[K, X](x)
    clipping match {
      case Some(thr) ⇒
        val M = x.valuesIterator.map(Math.abs).max
        val sqSize = Math.sqrt(x.size)

        // If M * sqrt(size) < threshold, norm < threshold.
        // Because norm = sqrt(sum(x^2)) < sqrt(M^2 * size)
        if (M < thr / sqSize) {
          x
        } else {
          // Divide by Max value first, to avoid overflow error.
          val nX = mul(x, 1 / M)
          val n = normImpl(nX)

          // Original clipping was multiplying threshold/norm.
          // Since we normalized, divide threshold by M.
          val t = thr / M
          if (n >= t) {
            mul(nX, t / n)
          } else {
            x
          }
        }
      case _ ⇒
        x
    }
  }

  def clearInvalid[K, X <: Tensor[K, Double]](x: X, resetNaN: Double = 0.0) = {
    x.keySet.par.foreach { key ⇒
      val d = x(key)
      if (d.isNaN) x.update(key, resetNaN)
      else if (d.isInfinity) x.update(key, Double.MaxValue * d.signum)
    }
  }

  def scaleCheck(x: DataVec) = check[Int, DataVec](x)

  def scaleCheck(x: Matrix) = check[(Int, Int), Matrix](x)

  /**
   * Set scaling down threshold.
   * @param thr Threshold for scalind down.
   */
  def scalingDownBy(thr: Double) = {
    clipping = Some(thr)
  }
}

/**
 * Class for Weight object.
 * @tparam X Either Matrix or DataVec
 */
class Weight[X <: Tensor[_, Double]] extends Serializable with KryoSerializable {
  /** Update algorithm **/
  @transient private var _updater: Algorithm[X] = _
  /** Value **/
  private var _x: X = null.asInstanceOf[X]

  /**
   * Check if the weight is successfully defined.
   * @return True if weight has value.
   */
  def isDefined = _x != null

  /**
   * Compute weight loss
   * @param normImpl Norm implementation (implicit)
   * @return Weight Loss (double).
   */
  def loss(implicit normImpl: norm.Impl[X, Double]): Double =
    if (_updater != null) {
      val n = normImpl(_x)
      n * n * _updater.l2factor / 2.0
    } else {
      throw new IllegalStateException("Weight instance does not have any Algorithm instance")
    }

  /**
   * Execute update algorithm.
   * @param seq Sequence of delta weights, per each instance.
   */
  def update(seq: collection.GenSeq[X])(implicit add: OpAdd.InPlaceImpl2[X, X]): Unit =
    if (_updater != null) {
      if (seq.nonEmpty) {
        val delta = seq.reduce[X] { case (x, y) ⇒
          add(x, y)
          x
        }
        _updater.update(delta)
      }
    } else {
      throw new IllegalStateException("Weight instance does not have any Algorithm instance")
    }

  /**
   * Value of this weight.
   * @return value of this weight.
   */
  def value = _x

  /**
   * Set update algorithm for this weight.
   * @param updater Update algorithm
   * @return self.
   */
  def withUpdater(updater: Algorithm[X]) = {
    _updater = updater
    this
  }

  /**
   * Set value for this weight.
   * @param x Value
   * @return self.
   */
  def withValue(x: X) = {
    _x = x
    this
  }

  override def read(kryo: Kryo, input: Input): Unit = {
    _x = kryo.readClassAndObject(input).asInstanceOf[X]
  }

  override def write(kryo: Kryo, output: Output): Unit = {
    kryo.writeClassAndObject(output, _x)
  }
}

/**
 * __Trait__ for update algorithm
 * @tparam X Either Matrix or DataVec
 */
trait Algorithm[X <: QuasiTensor[_, Double]] {
  /** L2 regularization factor **/
  val l2factor: Double
  /** Value **/
  protected val x: X

  /**
   * Execute the algorithm for given __Δweight__ and __weights__
   * @param dW delta weight.
   */
  def update(dW: X): Unit
}

/**
 * __Builder__ for AdaDelta algorithm
 *
 * If you are trying to use this algorithm for your research,
 * you should add a reference to [[http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf AdaDelta techinical report]].
 *
 * @param l2decay L,,2,, regularization factor `(Default 0.0001)`
 * @param historyDecay AdaDelta history decay factor `(Default 95% = 0.95)`
 * @param historyEpsilon AdaDelta base factor `(Default 1e-6)`
 *
 * @example {{{val algorithm = new AdaDelta(l2decay = 0.0001)}}}
 */
class AdaDelta(var l2decay: Double = 0.0001,
               var historyDecay: Double = 0.95,
               var historyEpsilon: Double = 1e-6)
  extends WeightBuilder {
  def this() = this(0.0001, 0.95, 1e-6)

  override def read(kryo: Kryo, input: Input): Unit = {
    l2decay = input.readDouble()
    historyDecay = input.readDouble()
    historyEpsilon = input.readDouble()
  }

  override def write(kryo: Kryo, output: Output): Unit = {
    output.writeDouble(l2decay)
    output.writeDouble(historyDecay)
    output.writeDouble(historyEpsilon)
  }

  override protected def getUpdater(value: Matrix, noReg: Boolean): Algorithm[Matrix] =
    new Updater[(Int, Int), Matrix](value, if (noReg) 0.0 else l2decay)

  override protected def getUpdater(value: DataVec, noReg: Boolean): Algorithm[DataVec] =
    new Updater[Int, DataVec](value, if (noReg) 0.0 else l2decay)

  /**
   * Algorithm implementation.
   * @param x Value
   * @param l2factor L,,2,, Regularization factor
   * @param zero Zero object of value (implicit)
   * @param elemMul Scalar Multiplication implementation (implicit)
   * @param mul Multiplication implementation (implicit)
   * @param mulScalarInplace Inplace Scalar Multiplication implementation (implicit)
   * @param addInplace Inplace addition implementation (implicit)
   * @param addScalar scalar addition implementation (implicit)
   * @param root Square root implementation (implicit)
   * @param subInplace Inplace subtraction implementation (implicit)
   * @param elemDiv Element-wise division implementation (implicit)
   * @tparam X Either Matrix or DataVec
   */
  class Updater[K, X <: Tensor[K, Double]](override protected val x: X,
                                           override val l2factor: Double)
                                          (implicit private val zero: Zero[X],
                                           implicit private val elemMul: OpMulScalar.Impl2[X, X, X],
                                           implicit private val mul: OpMulScalar.Impl2[X, Double, X],
                                           implicit private val mulScalarInplace: OpMulScalar.InPlaceImpl2[X, Double],
                                           implicit private val addInplace: OpAdd.InPlaceImpl2[X, X],
                                           implicit private val addScalar: OpAdd.Impl2[X, Double, X],
                                           implicit private val root: sqrt.Impl[X, X],
                                           implicit private val subInplace: OpSub.InPlaceImpl2[X, X],
                                           implicit private val elemDiv: OpDiv.Impl2[X, X, X],
                                           implicit private val normImpl: norm.Impl[X, Double])
    extends Algorithm[X] {
    /** Update history **/
    private lazy val deltaSq = zero(x)
    /** Gradient history **/
    private lazy val gradSq = zero(x)

    override def update(dW: X): Unit = {
      val d = mul(x, l2decay)
      addInplace(d, dW)

      mulScalarInplace(gradSq, historyDecay)
      val decayD = elemMul(d, d)
      addInplace(gradSq, mul(decayD, 1.0 - historyDecay))

      val r1 = root(addScalar(deltaSq, historyEpsilon))
      val r2 = root(addScalar(gradSq, historyEpsilon))
      val rate = elemDiv(r1, r2)

      val dw = elemMul(d, rate)
      subInplace(x, dw)
      Weight.clearInvalid[K, X](x)

      mulScalarInplace(deltaSq, historyDecay)
      val decayDW = elemMul(dw, dw)
      addInplace(deltaSq, mul(decayDW, 1.0 - historyDecay))
    }
  }

}

/**
 * __Builder__ for AdaGrad algorithm.
 *
 * If you are trying to use this algorithm for your research, you should add a reference to [[http://www.magicbroom.info/Papers/DuchiHaSi10.pdf AdaGrad paper]].
 *
 * @param rate the learning rate `(Default 0.6)`
 * @param l2decay L,,2,, regularization factor `(Default 0.0001)`
 * @param fudgeFactor Factor for smoothing effect `(Default 1e-6)`
 *
 * @example {{{val algorithm = new AdaGrad(l2decay = 0.0001)}}}
 */
class AdaGrad(var rate: Double = 0.6,
              var l2decay: Double = 0.0001,
              var fudgeFactor: Double = 1e-6)
  extends WeightBuilder {
  def this() = this(0.6, 0.0001, 1e-6)

  override def read(kryo: Kryo, input: Input): Unit = {
    l2decay = input.readDouble()
    rate = input.readDouble()
    fudgeFactor = input.readDouble()
  }

  override def write(kryo: Kryo, output: Output): Unit = {
    output.writeDouble(l2decay)
    output.writeDouble(rate)
    output.writeDouble(fudgeFactor)
  }

  override protected def getUpdater(value: Matrix, noReg: Boolean): Algorithm[Matrix] =
    new Updater[(Int, Int), Matrix](value, if (noReg) 0.0 else l2decay)

  override protected def getUpdater(value: DataVec, noReg: Boolean): Algorithm[DataVec] =
    new Updater[Int, DataVec](value, if (noReg) 0.0 else l2decay)

  /**
   * Algorithm implementation.
   * @param x Value
   * @param l2factor L,,2,, Regularization factor
   * @param zero Zero object of value (implicit)
   * @param elemMul Scalar Multiplication implementation (implicit)
   * @param mul Multiplication implementation (implicit)
   * @param add Addition implementation (implicit)
   * @param addInplace Inplace addition implementation (implicit)
   * @param root Square root implementation (implicit)
   * @param subInplace Inplace subtraction implementation (implicit)
   * @param div Scalar division implementation (implicit)
   * @tparam X Either Matrix or DataVec
   */
  class Updater[K, X <: Tensor[K, Double]](override protected val x: X,
                                           override val l2factor: Double)
                                          (implicit private val zero: Zero[X],
                                           implicit private val elemMul: OpMulScalar.Impl2[X, X, X],
                                           implicit private val mul: OpMulScalar.Impl2[X, Double, X],
                                           implicit private val add: OpAdd.Impl2[X, Double, X],
                                           implicit private val addInplace: OpAdd.InPlaceImpl2[X, X],
                                           implicit private val root: sqrt.Impl[X, X],
                                           implicit private val subInplace: OpSub.InPlaceImpl2[X, X],
                                           implicit private val div: OpDiv.Impl2[Double, X, X],
                                           implicit private val normImpl: norm.Impl[X, Double])
    extends Algorithm[X] {
    /** accumulated history of parameter updates */
    private lazy val history = zero(x)

    /**
     * Execute the algorithm for given __Δweight__ and __weights__
     */
    override def update(dW: X): Unit = {
      val d = mul(x, l2decay)
      addInplace(d, dW)
      addInplace(history, elemMul(d, d))

      val dw = elemMul(d, div(rate, add(root(history), fudgeFactor)))
      subInplace(x, dw)
      Weight.clearInvalid[K, X](x)
    }
  }
}

/**
 * __Builder__ for Stochastic Gradient Descent
 *
 * Basic Gradient Descent rule with mini-batch training.
 *
 * @param rate the learning rate `(Default 0.03)`
 * @param l2decay L,,2,, regularization factor `(Default 0.0001)`
 * @param momentum Momentum factor for adaptive learning `(Default 0.0001)`
 *
 * @example {{{val algorithm = new StochasticGradientDescent(l2decay = 0.0001)}}}
 */
class StochasticGradientDescent(var rate: Double = 0.03,
                                var l2decay: Double = 0.0001,
                                var momentum: Double = 0.0001)
  extends WeightBuilder {
  def this() = this(0.03, 0.0001, 0.0001)

  override def read(kryo: Kryo, input: Input): Unit = {
    l2decay = input.readDouble()
    rate = input.readDouble()
    momentum = input.readDouble()
  }

  override def write(kryo: Kryo, output: Output): Unit = {
    output.writeDouble(l2decay)
    output.writeDouble(rate)
    output.writeDouble(momentum)
  }

  override protected def getUpdater(value: Matrix, noReg: Boolean): Algorithm[Matrix] =
    new Updater[(Int, Int), Matrix](value, if (noReg) 0.0 else l2decay)

  override protected def getUpdater(value: DataVec, noReg: Boolean): Algorithm[DataVec] =
    new Updater[Int, DataVec](value, if (noReg) 0.0 else l2decay)

  /**
   * Algorithm implementation.
   * @param x Value
   * @param l2factor L,,2,, Regularization factor
   * @param zero Zero object of value (implicit)
   * @param mul Multiplication implementation (implicit)
   * @param mulInplace Inplace Scalar Multiplication implementation (implicit)
   * @param addInplace Inplace addition implementation (implicit)
   * @param subInplace Inplace subtraction implementation (implicit)
   * @tparam X Either Matrix or DataVec
   */
  class Updater[K, X <: Tensor[K, Double]](override protected val x: X,
                                           override val l2factor: Double)
                                          (implicit private val zero: Zero[X],
                                           implicit private val mul: OpMulScalar.Impl2[X, Double, X],
                                           implicit private val mulInplace: OpMulScalar.InPlaceImpl2[X, Double],
                                           implicit private val addInplace: OpAdd.InPlaceImpl2[X, X],
                                           implicit private val subInplace: OpSub.InPlaceImpl2[X, X],
                                           implicit private val normImpl: norm.Impl[X, Double])
    extends Algorithm[X] {
    /** the last update of parameters */
    private lazy val lastDelta: X = if (momentum != 0) zero(x) else null.asInstanceOf[X]

    /**
     * Execute the algorithm for given __Δweight__ and __weights__
     */
    def update(dW: X): Unit = {
      val d = mul(x, l2decay)
      addInplace(d, dW)
      if (lastDelta != null) {
        mulInplace(lastDelta, momentum)
        addInplace(lastDelta, d)
        subInplace(x, lastDelta)
      } else
        subInplace(x, d)
      Weight.clearInvalid[K, X](x)
    }
  }
}