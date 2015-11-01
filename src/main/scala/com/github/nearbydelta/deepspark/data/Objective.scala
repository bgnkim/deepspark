package com.github.nearbydelta.deepspark.data

import breeze.linalg.{DenseVector, sum}
import breeze.numerics._

/**
 * __Trait__ that describes an objective function for '''entire network'''
 *
 * Because these objective functions can be shared, we recommend to make inherited one as an object. 
 */
trait Objective extends ((DataVec, DataVec) ⇒ Double) with Serializable {
  /**
   * Compute differentiation value of this objective function at `x = r - o`
   *
   * @param real the expected __real output__, `r`
   * @param output the computed __output of the network__, `o`
   * @return differentiation value at `f(x)=fx`, which is __a column vector__
   */
  def derivative(real: DataVec, output: DataVec): DataVec

  /**
   * Compute error (loss)
   *
   * @param real the expected __real output__
   * @param output the computed __output of the network__
   * @return the error
   */
  override def apply(real: DataVec, output: DataVec): Double
}


/**
 * __Objective Function__: Cosine Similarity Error
 *
 * @note This function returns 1 - cosine similarity, i.e. cosine dissimiarlity.
 *
 * @example
 * {{{val output = net(input)
 *            val err = CosineErr(real, output)
 *            val diff = CosineErr.derivative(real, output)
 * }}}
 */
object CosineErr extends Objective {
  /**
   * length of given matrix
   * @param matrix matrix
   * @return length = sqrt(sum(pow(:, 2)))
   */
  private def len(matrix: DataVec): Double = {
    Math.sqrt(sum(pow(matrix, 2.0))).toFloat
  }

  override def apply(real: DataVec, output: DataVec): Double = {
    val norm = len(real) * len(output)
    val dotValue: Double = sum(real :* output)
    1.0 - (dotValue / norm)
  }

  override def derivative(real: DataVec, output: DataVec): DataVec = {
    val dotValue: Double = sum(real :* output)

    val lenReal = len(real)
    val lenOut = len(output)
    val lenOutSq = lenOut * lenOut
    // Denominator of derivative is len(real) * len(output)^3
    val denominator = lenReal * lenOut * lenOutSq

    DenseVector.tabulate(output.length) {
      r ⇒
        val x = output(r)
        val a = real(r)
        // The nominator of derivative of cosine similarity is,
        // a(lenOut^2 - x^2) - x(dot - a*x)
        // = a*lenOut^2 - x*dot
        val nominator = a * lenOutSq - x * dotValue

        // We need derivative of 1 - cosine.
        -(nominator / denominator)
    }
  }
}

/**
 * __Objective Function__: Sum of Cross-Entropy (Logistic)
 *
 * @note This objective function prefer 0/1 output
 * @example
 * {{{val output = net(input)
 *            val err = CrossEntropyErr(real, output)
 *            val diff = CrossEntropyErr.derivative(real, output)
 * }}}
 */
object CrossEntropyErr extends Objective {
  /**
   * Entropy function
   */
  val entropy = (r: Double, o: Double) ⇒
    (if (r != 0.0) -r * Math.log(o).toFloat else 0.0) + (if (r != 1.0) -(1.0 - r) * Math.log(1.0 - o).toFloat else 0.0)

  /**
   * Derivative of Entropy function
   */
  val entropyDiff = (r: Double, o: Double) ⇒ (r - o) / (o * (o - 1.0))

  override def apply(real: DataVec, output: DataVec): Double =
    sum(DenseVector.tabulate(real.length)(r ⇒ entropy(real(r), output(r))))

  override def derivative(real: DataVec, output: DataVec): DataVec =
    DenseVector.tabulate(real.length)(r ⇒ entropyDiff(real(r), output(r)))
}

/**
 * __Objective Function__: Sum of Squared Error
 *
 * @example
 * {{{val output = net(input)
 *            val err = SquaredErr(real, output)
 *            val diff = SquaredErr.derivative(real, output)
 * }}}
 */
object SquaredErr extends Objective {
  override def apply(real: DataVec, output: DataVec): Double = {
    val diff = real - output
    sum(pow(diff, 2.0))
  }

  override def derivative(real: DataVec, output: DataVec): DataVec = output - real
}

/**
 * __Objective Function__: Sum of Absolute Error
 *
 * @note In mathematics, L,,1,,-distance is called ''Manhattan distance.''
 *
 * @example
 * {{{val output = net(input)
 *             val err = ManhattanErr(real, output)
 *             val diff = ManhattanErr.derivative(real, output)
 * }}}
 */
object ManhattanErr extends Objective {
  override def apply(real: DataVec, output: DataVec): Double = {
    val diff = real - output
    sum(abs(diff))
  }

  override def derivative(real: DataVec, output: DataVec): DataVec =
    DenseVector.tabulate(real.length) {
      r ⇒
        val target = real(r)
        val x = output(r)
        if (target > x) 1.0
        else if (target < x) -1.0
        else 0.0
    }
}