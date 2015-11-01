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
   * @param delta A variable for storing change amount.
   * @param noReg True if this value does not require regularization.
   * @return Algorithm object
   */
  protected def getUpdater(value: Matrix, delta: Matrix, noReg: Boolean): Algorithm[Matrix]

  /**
   * Get Update Algorithm for given value.
   * @param value Target of alorithm.
   * @param delta A variable for storing change amount.
   * @param noReg True if this value does not require regularization.
   * @return Algorithm object
   */
  protected def getUpdater(value: DataVec, delta: DataVec, noReg: Boolean): Algorithm[DataVec]

  /**
   * Build weight update algorithm for given Weight object.
   * @param weight Weight object of Matrix
   * @param value target Matrix
   * @param noReg True if this weight does not require regularization.
   * @return Algorithm attached Weight object.
   */
  private def buildTo(weight: Weight[Matrix], value: Matrix, noReg: Boolean): Weight[Matrix] = {
    weight withValue value
    weight withUpdater getUpdater(value, weight.getDelta, noReg)
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
    weight withUpdater getUpdater(value, weight.getDelta, noReg)
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
   * @param normImpl Norm implementation (implicit)
   * @param mul Multiplication of Scalar inplace (implicit)
   * @tparam X Either Matrix or DataVec
   * @return Scaled down value.
   */
  def scaleCheck[X](x: X)
                   (implicit normImpl: norm.Impl[X, Double], mul: OpMulScalar.InPlaceImpl2[X, Double]): X =
    clipping match {
      case Some(thr) ⇒
        val n = normImpl(x)
        if (n >= thr) {
          mul(x, thr / n)
        }
        x
      case _ ⇒
        x
    }

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
class Weight[X] extends Serializable with KryoSerializable {
  /** Store for update amount. **/
  @transient private var _delta: X = _
  /** Update algorithm **/
  @transient private var _updater: Algorithm[X] = _
  /** Value **/
  private var _x: X = null.asInstanceOf[X]

  /**
   * Get accumulated update amount.
   * @return Accumualted update amount
   */
  def getDelta = _delta

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
      n * n * _updater.l2factor
    } else {
      throw new IllegalStateException("Weight instance does not have any Algorithm instance")
    }

  /**
   * Execute update algorithm.
   * @param count Count of minibatch instances.
   */
  def update(count: Int): Unit =
    if (_updater != null) {
      _updater.update(count)
    } else {
      throw new IllegalStateException("Weight instance does not have any Algorithm instance")
    }

  /**
   * Update this weight by given amount.
   * @param x amount of update, i.e. gradient.
   * @param param$1 Norm implementation (implicit)
   * @param param$2 Multiplication of Scalar inplace implementation (implicit)
   * @param addInplace Add in place implementation (implicit)
   */
  def updateBy(x: X)(implicit param$1: norm.Impl[X, Double], param$2: OpMulScalar.InPlaceImpl2[X, Double],
                     addInplace: OpAdd.InPlaceImpl2[X, X]): Unit = {
    // Use Scaling Down Trick (Pascanu et al., 2013)
    addInplace(_delta, Weight.scaleCheck(x))
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
   * @param zero Zero object for this value. (implicit)
   * @return self.
   */
  def withValue(x: X)(implicit zero: Zero[X]) = {
    _x = x
    _delta = zero(x)
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
trait Algorithm[X] {
  /** L2 regularization factor **/
  val l2factor: Double
  /** Value **/
  protected val x: X
  /** Accumulated update amount **/
  protected val dW: X

  /**
   * Execute the algorithm for given __Δweight__ and __weights__
   */
  def update(count: Int): Unit
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

  override protected def getUpdater(value: Matrix, delta: Matrix, noReg: Boolean): Algorithm[Matrix] =
    new Updater[Matrix](value, delta, if (noReg) 0.0 else l2decay)

  override protected def getUpdater(value: DataVec, delta: DataVec, noReg: Boolean): Algorithm[DataVec] =
    new Updater[DataVec](value, delta, if (noReg) 0.0 else l2decay)

  /**
   * Algorithm implementation.
   * @param x Value
   * @param dW Store for update amount.
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
   * @param set Assign implementation (implicit)
   * @tparam X Either Matrix or DataVec
   */
  class Updater[X](override protected val x: X,
                   override protected val dW: X,
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
                   implicit private val set: OpSet.InPlaceImpl2[X, Double]) extends Algorithm[X] {
    /** Update history **/
    private lazy val deltaSq = zero(x)
    /** Gradient history **/
    private lazy val gradSq = zero(x)

    /**
     * Execute the algorithm for given __Δ(weight)__ and __weights__
     */
    override def update(count: Int): Unit = {
      val d = mul(x, l2decay * 2)
      mulScalarInplace(dW, 1.0 / count)
      addInplace(d, dW)

      mulScalarInplace(gradSq, historyDecay)
      val decayD = elemMul(d, d)
      addInplace(gradSq, mul(decayD, 1.0 - historyDecay))

      val r1 = root(addScalar(deltaSq, historyEpsilon))
      val r2 = root(addScalar(gradSq, historyEpsilon))
      val rate = elemDiv(r1, r2)

      val dw = elemMul(d, rate)
      subInplace(x, dw)

      mulScalarInplace(deltaSq, historyDecay)
      val decayDW = elemMul(dw, dw)
      addInplace(deltaSq, mul(decayDW, 1.0 - historyDecay))

      set(dW, 0.0)
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

  override protected def getUpdater(value: Matrix, delta: Matrix, noReg: Boolean): Algorithm[Matrix] =
    new Updater[Matrix](value, delta, if (noReg) 0.0 else l2decay)

  override protected def getUpdater(value: DataVec, delta: DataVec, noReg: Boolean): Algorithm[DataVec] =
    new Updater[DataVec](value, delta, if (noReg) 0.0 else l2decay)

  /**
   * Algorithm implementation.
   * @param x Value
   * @param dW Store for update amount.
   * @param l2factor L,,2,, Regularization factor
   * @param zero Zero object of value (implicit)
   * @param elemMul Scalar Multiplication implementation (implicit)
   * @param mul Multiplication implementation (implicit)
   * @param add Addition implementation (implicit)
   * @param addInplace Inplace addition implementation (implicit)
   * @param root Square root implementation (implicit)
   * @param subInplace Inplace subtraction implementation (implicit)
   * @param div Scalar division implementation (implicit)
   * @param set Assign implementation (implicit)
   * @tparam X Either Matrix or DataVec
   */
  class Updater[X](override protected val x: X,
                   override protected val dW: X,
                   override val l2factor: Double)
                  (implicit private val zero: Zero[X],
                   implicit private val elemMul: OpMulScalar.Impl2[X, X, X],
                   implicit private val mul: OpMulScalar.Impl2[X, Double, X],
                   implicit private val add: OpAdd.Impl2[X, Double, X],
                   implicit private val addInplace: OpAdd.InPlaceImpl2[X, X],
                   implicit private val root: sqrt.Impl[X, X],
                   implicit private val subInplace: OpSub.InPlaceImpl2[X, X],
                   implicit private val div: OpDiv.Impl2[Double, X, X],
                   implicit private val set: OpSet.InPlaceImpl2[X, Double]) extends Algorithm[X] {
    /** accumulated history of parameter updates */
    private lazy val history = zero(x)

    /**
     * Execute the algorithm for given __Δweight__ and __weights__
     */
    override def update(count: Int): Unit = {
      val d = mul(x, l2decay * 2)
      addInplace(d, mul(dW, 1.0 / count))

      addInplace(history, elemMul(d, d))

      val dw = elemMul(d, div(rate, add(root(history), fudgeFactor)))
      subInplace(x, dw)

      set(dW, 0.0)
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

  override protected def getUpdater(value: Matrix, delta: Matrix, noReg: Boolean): Algorithm[Matrix] =
    new Updater[Matrix](value, delta, if (noReg) 0.0 else l2decay)

  override protected def getUpdater(value: DataVec, delta: DataVec, noReg: Boolean): Algorithm[DataVec] =
    new Updater[DataVec](value, delta, if (noReg) 0.0 else l2decay)

  /**
   * Algorithm implementation.
   * @param x Value
   * @param dW Store for update amount.
   * @param l2factor L,,2,, Regularization factor
   * @param zero Zero object of value (implicit)
   * @param mul Multiplication implementation (implicit)
   * @param mulInplace Inplace Scalar Multiplication implementation (implicit)
   * @param addInplace Inplace addition implementation (implicit)
   * @param subInplace Inplace subtraction implementation (implicit)
   * @param set Assign implementation (implicit)
   * @tparam X Either Matrix or DataVec
   */
  class Updater[X](override protected val x: X,
                   override protected val dW: X,
                   override val l2factor: Double)
                  (implicit private val zero: Zero[X],
                   implicit private val mul: OpMulScalar.Impl2[X, Double, X],
                   implicit private val mulInplace: OpMulScalar.InPlaceImpl2[X, Double],
                   implicit private val addInplace: OpAdd.InPlaceImpl2[X, X],
                   implicit private val subInplace: OpSub.InPlaceImpl2[X, X],
                   implicit private val set: OpSet.InPlaceImpl2[X, Double]) extends Algorithm[X] {
    /** the last update of parameters */
    private lazy val lastDelta: X = if (momentum != 0) zero(x) else null.asInstanceOf[X]

    /**
     * Execute the algorithm for given __Δweight__ and __weights__
     */
    def update(count: Int): Unit = {
      mulInplace(dW, 1.0 / count)

      val d = mul(x, l2decay * 2)
      addInplace(d, dW)
      if (lastDelta != null) {
        mulInplace(lastDelta, momentum)
        addInplace(lastDelta, d)
        subInplace(x, lastDelta)
      } else
        subInplace(x, d)

      set(dW, 0.0)
    }
  }
}