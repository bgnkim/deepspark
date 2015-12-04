package com.github.nearbydelta.deepspark.word

import breeze.linalg._
import breeze.numerics._
import com.esotericsoftware.kryo.io.{Input, Output}
import com.esotericsoftware.kryo.{Kryo, KryoSerializable}
import com.github.nearbydelta.deepspark.data.{DataVec, Weight}

import scala.collection.mutable

/**
 * __Trait__ that describes the algorithm for weight update
 *
 * Because each weight update requires history, we recommend to make inherited one as a class. 
 */
trait LedgerBuilder extends Serializable with KryoSerializable {
  def getUpdater(model: ListLedger): LedgerAlgorithm
}

trait LedgerAlgorithm {
  protected lazy val dimension = x.head.length
  val delta: LedgerNote = new mutable.HashMap[Int, DataVec]() with mutable.SynchronizedMap[Int, DataVec]
  val x: ListLedger

  def loss: Double = x.map { v ⇒
    val n = norm(v)
    n * n
  }.sum[Double]

  /**
   * Execute the algorithm for given __Δweight__ and __weights__
   */
  def update: () ⇒ Unit
}


/**
 * __Algorithm__: AdaDelta algorithm
 *
 * If you are trying to use this algorithm for your research, you should add a reference to [[http://www.matthewzeiler.com/pubs/googleDataVecR2012/googleDataVecR2012.pdf AdaDelta techinical report]].
 *
 * @param l2decay L,,2,, regularization factor `(Default 0.0001)`
 * @param historyDecay AdaDelta history decay factor `(Default 95% = 0.95)`
 * @param historyEpsilon AdaDelta base factor `(Default 1e-6)`
 *
 * @example {{{val algorithm = new AdaDelta(l2decay = 0.0001)}}}
 */
class LedgerAdaDelta(var l2decay: Double = 0.0001,
                     var historyDecay: Double = 0.95,
                     var historyEpsilon: Double = 1e-6)
  extends LedgerBuilder {
  def this() = this(0.0001, 0.95, 1e-6)

  override def getUpdater(model: ListLedger): LedgerAlgorithm =
    new Updater(model)

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

  class Updater(override val x: ListLedger)
    extends LedgerAlgorithm {
    private lazy val deltaSq = LedgerNoteZero() withDefault (_ ⇒ DenseVector.zeros[Double](dimension))
    private lazy val gradSq = LedgerNoteZero() withDefault (_ ⇒ DenseVector.zeros[Double](dimension))

    override def loss = l2decay * super.loss / 2.0

    /**
     * Execute the algorithm for given __Δ(weight)__ and __weights__
     */
    override def update = () ⇒ {
      delta.par.foreach {
        case (id, dW) ⇒
          val vec = x(id)

          axpy(l2decay, vec, dW)
          Weight.check[Int, DataVec](dW)

          val gSq = gradSq(id)
          val dSq = deltaSq(id)

          gSq *= historyDecay
          val decayD = dW :* dW
          axpy(1.0 - historyDecay, decayD, gSq)

          val dw = dSq.mapPairs {
            case (i, ds) ⇒
              val dsq = ds + historyEpsilon
              val gsq = gSq(i) + historyEpsilon
              dW(i) * sqrt(dsq / gsq)
          }

          vec -= dw
          Weight.clearInvalid[Int, DataVec](vec)

          dSq *= historyDecay
          val decayDW = dw :* dw
          axpy(1.0 - historyDecay, decayDW, dSq)
        case _ ⇒
      }
      delta.clear()
    }
  }

}

/**
 * __Algorithm__: AdaGrad algorithm.
 *
 * If you are trying to use this algorithm for your research, you should add a reference to [[http://www.magicbroom.info/Papers/DuchiHaSi10.pdf AdaGrad paper]].
 *
 * @param rate the learning rate `(Default 0.6)`
 * @param l2decay L,,2,, regularization factor `(Default 0.0001)`
 *
 *
 * @example {{{val algorithm = new AdaGrad(l2decay = 0.0001)}}}
 */
class LedgerAdaGrad(var rate: Double = 0.6,
                    var l2decay: Double = 0.0001,
                    var fudgeFactor: Double = 1e-6)
  extends LedgerBuilder {
  def this() = this(0.6, 0.0001, 1e-6)

  override def getUpdater(model: ListLedger): LedgerAlgorithm =
    new Updater(model)

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

  class Updater(override val x: ListLedger)
    extends LedgerAlgorithm {
    private lazy val history = LedgerNoteZero() withDefault (_ ⇒ DenseVector.zeros[Double](dimension))

    override def loss = l2decay * super.loss / 2.0

    /**
     * Execute the algorithm for given __Δ(weight)__ and __weights__
     */
    override def update = () ⇒ {
      delta.par.foreach {
        case (id, dW) ⇒
          val vec = x(id)

          axpy(l2decay, vec, dW)
          Weight.check[Int, DataVec](dW)

          val hW = history(id)

          hW += (dW :* dW)

          val dw = hW.mapPairs {
            case (i, h) ⇒
              dW(i) * rate / (sqrt(h) + fudgeFactor)
          }

          vec -= dw
          Weight.clearInvalid[Int, DataVec](vec)
        case _ ⇒
      }
      delta.clear()
    }
  }

}

/**
 * __Algorithm__: Stochastic Gradient Descent
 *
 * Basic Gradient Descent rule with mini-batch training.
 *
 * @param rate the learning rate `(Default 0.03)`
 * @param l2decay L,,2,, regularization factor `(Default 0.0001)`
 * @param momentum Momentum factor for adaptive learning `(Default 0.0001)`
 *
 * @example {{{val algorithm = new StochasticGradientDescent(l2decay = 0.0001)}}}
 */
class LedgerSGD(var rate: Double = 0.03,
                var l2decay: Double = 0.0001,
                var momentum: Double = 0.0001)
  extends LedgerBuilder {
  def this() = this(0.03, 0.0001, 0.0001)

  override def getUpdater(model: ListLedger): LedgerAlgorithm =
    new Updater(model)

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

  class Updater(override val x: ListLedger)
    extends LedgerAlgorithm {
    @transient private lazy val history = LedgerNoteZero() withDefault (_ ⇒ DenseVector.zeros[Double](dimension))

    override def loss = l2decay * super.loss / 2.0

    /**
     * Execute the algorithm for given __Δ(weight)__ and __weights__
     */
    override def update = () ⇒ {
      delta.par.foreach {
        case (id, dW) ⇒
          val vec = x(id)

          axpy(l2decay, vec, dW)
          Weight.check[Int, DataVec](dW)

          val value =
            if (momentum != 0.0) {
              val hW = history(id)
              hW *= momentum
              hW -= dW
              hW
            } else {
              dW
            }

          vec -= value
          Weight.clearInvalid[Int, DataVec](vec)
        case _ ⇒
      }
      delta.clear()
    }
  }

}
