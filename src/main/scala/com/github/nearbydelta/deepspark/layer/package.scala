package com.github.nearbydelta.deepspark

import com.esotericsoftware.kryo.io.{Input, Output}
import com.esotericsoftware.kryo.{Kryo, KryoSerializable}
import com.github.nearbydelta.deepspark.data.{DataVec, WeightBuilder}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.parallel.ParSeq
import scala.reflect.{ClassTag, classTag}

/**
 * Created by bydelta on 15. 10. 16.
 */
package object layer {

  trait Layer[In, OutInfo] extends Serializable with KryoSerializable {
    @transient implicit protected val evidenceI: ClassTag[In]
    var NIn: Int = 0
    var NOut: Int = 0
    val outVecOf: OutInfo ⇒ DataVec
    @transient protected var isUpdatable: Boolean = false
    @transient protected var inoutSEQ: Seq[(In, OutInfo)] = null

    def apply(in: In): OutInfo

    final def forward(in: In): DataVec = outVecOf(apply(in))

    def backward(in: In, out: OutInfo, error: DataVec): DataVec

    def withInput(in: Int): this.type = {
      NIn = in
      this
    }

    def withOutput(out: Int): this.type = {
      NOut = out
      this
    }

    def setUpdatable(bool: Boolean): Unit = {
      this.isUpdatable = bool
    }

    def apply(in: RDD[(Long, In)]): RDD[(Long, DataVec)] = {
      in.mapValues(x ⇒ outVecOf(apply(x)))
    }

    final def forward(in: RDD[(Long, In)]) = apply(in)

    def forward(in: ParSeq[In]): ParSeq[DataVec] = {
      val seq = in.map(x ⇒ (x, apply(x)))
      val output = seq.map { x ⇒
        val vec = outVecOf(x._2)
        require(vec.data.count(x ⇒ x.isNaN || x.isInfinity) == 0,
          s"${this.getClass}, $NIn -> $NOut has NaN/Infinity! (IN: ${x._1})")
        vec
      }

      if (isUpdatable)
        inoutSEQ = seq.seq

      output
    }

    def backward(error: Seq[DataVec]): Seq[DataVec] = {
      inoutSEQ.zip(error).map {
        case ((x, y), e) ⇒ backward(x, y, e)
      }
    }

    def loss: Double

    def update(count: Int): Unit

    override def write(kryo: Kryo, output: Output): Unit = {
      output.writeInt(NIn)
      output.writeInt(NOut)
    }

    override def read(kryo: Kryo, input: Input): Unit = {
      NIn = input.readInt
      NOut = input.readInt
    }
  }

  trait TransformLayer extends Layer[DataVec, DataVec] {
    @transient implicit override protected val evidenceI: ClassTag[DataVec] = classTag[DataVec]
    override final val outVecOf = (x: DataVec) ⇒ x

    def initiateBy(builder: WeightBuilder): this.type
  }

  trait InputLayer[In, OutInfo] extends Layer[In, OutInfo]

}
