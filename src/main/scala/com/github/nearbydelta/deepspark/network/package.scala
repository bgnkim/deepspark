package com.github.nearbydelta.deepspark

import com.esotericsoftware.kryo.KryoSerializable
import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.layer._
import com.twitter.chill.{Input, Kryo, Output}
import org.apache.spark.rdd.RDD

import scala.annotation.tailrec
import scala.collection.mutable
import scala.collection.parallel.ParSeq
import scala.reflect.ClassTag
import scala.reflect.io.{File, Path}

/**
 * Created by bydelta on 15. 10. 17.
 */
package object network {

  trait Network[In, Out] extends Serializable with KryoSerializable {
    protected val layerSeq: mutable.ListBuffer[TransformLayer] = mutable.ListBuffer()
    @transient protected var builder: WeightBuilder = _

    def NOut =
      if (layerSeq.nonEmpty) layerSeq.last.NOut
      else 0

    def layers: Seq[TransformLayer] = layerSeq

    def saveTo(file: Path) =
      try {
        //        val oos = new ObjectOutputStream(File(file).outputStream())
        //        oos.writeObject(this)
        //        oos.close()
        val output = new Output(File(file).outputStream())
        KryoWrap.kryo.writeClassAndObject(output, this)
        output.close()
      } catch {
        case _: Throwable ⇒
      }

    def setUpdatable(bool: Boolean) = {
      layerSeq.foreach(_.setUpdatable(bool))
      this
    }

    def loss = layerSeq.par.map(_.loss).sum

    def update(count: Int): Unit = layerSeq.par.map { x ⇒ x.update(count); 0 }

    def initiateBy(builder: WeightBuilder) = {
      this.builder = builder
      this
    }

    final def add(layer: TransformLayer) = {
      require(builder != null, "Set weight builder before add layer!")
      val prevout = NOut
      if (prevout != 0 && layer.NIn != prevout) {
        layer.withInput(prevout)
      }
      layer.initiateBy(builder)
      layerSeq += layer
      this
    }

    @tailrec
    protected final def forwardRDD(in: RDD[(Long, DataVec)], layers: Seq[TransformLayer] = layerSeq): RDD[(Long, DataVec)] =
      if (layers.isEmpty) in
      else forwardRDD(layers.head forward in, layers.tail)

    @tailrec
    protected final def forwardSeq(in: ParSeq[DataVec], layers: Seq[TransformLayer] = layerSeq): ParSeq[DataVec] =
      if (layers.isEmpty) in
      else forwardSeq(layers.head forward in, layers.tail)

    @tailrec
    protected final def forwardSingle(in: DataVec, layers: Seq[TransformLayer] = layerSeq): DataVec =
      if (layers.isEmpty) in
      else forwardSingle(layers.head forward in, layers.tail)

    @tailrec
    protected final def backwardSeq(err: Seq[DataVec],
                                    layers: Seq[TransformLayer] = layerSeq.reverse): Seq[DataVec] =
      if (layers.isEmpty) err
      else backwardSeq(layers.head backward err, layers.tail)

    def forward(in: RDD[(Long, In)]): RDD[(Long, DataVec)]

    def forward(in: ParSeq[In]): ParSeq[DataVec]

    def forward(in: In): DataVec

    def backward(err: Seq[DataVec]): Unit

    def predictSoft(in: RDD[In])(implicit evidence$1: ClassTag[DataVec]): RDD[DataVec] = forward(in.map(0l → _)).map(_._2)

    def predictSoft(in: In): DataVec = forward(in)

    override def write(kryo: Kryo, output: Output): Unit = {
      kryo.writeClassAndObject(output, builder)
      output.writeInt(layerSeq.length)
      layerSeq.foreach(kryo.writeClassAndObject(output, _))
    }

    override def read(kryo: Kryo, input: Input): Unit = {
      builder = kryo.readClassAndObject(input).asInstanceOf[WeightBuilder]
      val size = input.readInt()
      layerSeq.clear()
      (0 until size).foreach { _ ⇒
        val layer = kryo.readClassAndObject(input).asInstanceOf[TransformLayer]
        layer.initiateBy(builder)
        layerSeq append layer
      }
    }
  }

  object Network {
    def readFrom[T <: Network[_, _]](file: Path) = {
      //      val ois = new ObjectInputStream(File(file).inputStream()) {
      //        override def resolveClass(desc: java.io.ObjectStreamClass): Class[_] = {
      //          try { Class.forName(desc.getName, false, getClass.getClassLoader) }
      //          catch { case ex: ClassNotFoundException => super.resolveClass(desc) }
      //        }
      //      }
      //      val model = ois.readObject().asInstanceOf[T]
      //      ois.close()
      //      model
      val input = new Input(File(file).inputStream())
      val obj = KryoWrap.kryo.readClassAndObject(input).asInstanceOf[T]
      input.close()
      obj
    }
  }

}
