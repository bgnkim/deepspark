package com.github.nearbydelta.deepspark

import com.esotericsoftware.kryo.io.{Input, Output}
import com.esotericsoftware.kryo.{Kryo, KryoSerializable}
import com.github.nearbydelta.deepspark.data.{DataVec, WeightBuilder}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.parallel.ParSeq
import scala.reflect.{ClassTag, classTag}

/**
 * Package deepspark.layer
 *
 * Package of layer classes.
 */
package object layer {

  /**
   * __Trait__ of layer.
   * @tparam In Input type
   * @tparam OutInfo Output type to store.
   */
  trait Layer[In, OutInfo] extends Serializable with KryoSerializable {
    /** Output converter from OutInfo to Vector **/
    val outVecOf: OutInfo ⇒ DataVec
    /** ClassTag for input **/
    @transient implicit protected val evidenceI: ClassTag[In]
    /** Size of input **/
    var NIn: Int = 0
    /** Size of output **/
    var NOut: Int = 0
    /** Sequence for backpropagation. Stores output values. **/
    @transient protected var inoutSEQ: Seq[(In, OutInfo)] = null
    /** True if this layer affected by backward propagation **/
    @transient protected var isUpdatable: Boolean = false

    /**
     * Apply this layer (forward computation)
     * @param in Input value
     * @return Output computation information.
     */
    def apply(in: In): OutInfo

    /**
     * Apply RDD of vector.
     * @param in RDD of (ID, Value), ID is Long value.
     * @return RDD of (ID, Vector), with same ID for input.
     */
    def apply(in: RDD[(Long, In)]): RDD[(Long, DataVec)] = {
      in.mapValues(x ⇒ outVecOf(apply(x)))
    }

    /**
     * Backward computation
     * @note <p>
     *       For the computation, we only used denominator layout. (cf. Wikipedia Page of Matrix Computation)
     *       For the computation rules, see "Matrix Cookbook" from MIT.
     *       </p>
     * @param seq Sequence of entries to be used for backward computation.
     * @return Error sequence, to backpropagate into previous layer.
     */
    def backward(seq: ParSeq[((In, OutInfo), DataVec)]): Seq[DataVec]

    /**
     * Backward computation using propagated error.
     * @param error Propagated error sequence.
     * @return Error sequence for back propagation.
     */
    def backward(error: Seq[DataVec]): Seq[DataVec] = backward(inoutSEQ.zip(error).par)

    /**
     * Apply this layer (forward computation)
     * @param in Input value
     * @return Output Vector
     */
    final def forward(in: In): DataVec = outVecOf(apply(in))

    /**
     * Apply RDD of vector.
     * @param in RDD of (ID, Value), ID is Long value.
     * @return RDD of (ID, Vector), with same ID for input.
     */
    final def forward(in: RDD[(Long, In)]) = apply(in)

    /**
     * Apply Parallel Sequence of vector.
     * @param in Parallel Sequence of Input
     * @return Parallel Sequence of Vector
     */
    def forward(in: ParSeq[In]): ParSeq[DataVec] = {
      val seq = in.map(x ⇒ (x, apply(x)))
      val output = seq.map { x ⇒
        val vec = outVecOf(x._2)
        if (vec != null) {
          val nan = vec.data.count(_.isNaN)
          val inf = vec.data.count(_.isInfinity)
          require(nan + inf == 0,
            s"${this.getClass}, $NIn -> $NOut has $nan NaN & $inf Infinity!")
        }
        vec
      }

      if (isUpdatable)
        inoutSEQ = seq.seq

      output
    }

    /**
     * Set weight builder for this layer.
     * @param builder Weight builder to be applied
     * @return self
     */
    def initiateBy(builder: WeightBuilder): this.type = this

    /**
     * Weight Loss of this layer
     * @return Weight loss
     */
    def loss: Double

    /**
     * Assign whether this layer updatable or not. value.
     * @param bool True if this layer used in backpropagation.
     */
    def setUpdatable(bool: Boolean): Unit = {
      this.isUpdatable = bool
    }

    /**
     * Set input size
     * @param in Size of input
     * @return self
     */
    def withInput(in: Int): this.type = {
      NIn = in
      this
    }

    /**
     * Set output size
     * @param out Size of output
     * @return self
     */
    def withOutput(out: Int): this.type = {
      NOut = out
      this
    }

    override def read(kryo: Kryo, input: Input): Unit = {
      NIn = input.readInt
      NOut = input.readInt
    }

    override def write(kryo: Kryo, output: Output): Unit = {
      output.writeInt(NIn)
      output.writeInt(NOut)
    }
  }

  /**
   * __Trait__ for hidden/output layer.
   */
  trait TransformLayer extends Layer[DataVec, DataVec] {
    override final val outVecOf = (x: DataVec) ⇒ x
    @transient implicit override protected val evidenceI: ClassTag[DataVec] = classTag[DataVec]
  }

  /**
   * __Trait__ for input layer.
   * @tparam In Input type
   * @tparam OutInfo Output type to store.
   */
  trait InputLayer[In, OutInfo] extends Layer[In, OutInfo] {
    /**
     * Broadcast resources of this layer.
     * @param sc Spark Context
     */
    def broadcast(sc: SparkContext): Unit

    /**
     * Unpersist broadcast of this layer.
     */
    def unbroadcast(): Unit
  }

}
