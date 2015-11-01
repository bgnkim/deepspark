package com.github.nearbydelta.deepspark.word

import breeze.linalg.DenseVector
import com.esotericsoftware.kryo.io.{Input, Output}
import com.esotericsoftware.kryo.{Kryo, KryoSerializable}
import com.github.nearbydelta.deepspark.data._
import org.apache.log4j.Logger

import scala.collection.mutable
import scala.io.Codec
import scala.reflect.io.{File, Path}

/**
 * Class for word list, or ledger words.
 */
class LedgerWords extends Serializable with KryoSerializable {
  /** ID for pad */
  lazy final val padID = words(LedgerModel.PAD)
  /** Size of this layer */
  lazy final val size = words.size
  /** ID for Unknown */
  lazy final val unkID = words(LedgerModel.UnkAll)
  /** Mapping of words */
  val words = mutable.HashMap[String, Int]()

  /**
   * Get index of given word
   * @param str Word
   * @return Index
   */
  def indexOf(str: String): Int = {
    words.get(str) match {
      case Some(id) ⇒ id
      case None ⇒
        words.get(str.getShape) match {
          case Some(id) ⇒ id
          case None ⇒ unkID
        }
    }
  }

  override def read(kryo: Kryo, input: Input): Unit = {
    words.clear()
    val size = input.readInt()
    (0 until size).foreach { _ ⇒
      val str = input.readString()
      val id = input.readInt()
      words += (str → id)
    }
  }

  // Please change word layer if change this.
  override def write(kryo: Kryo, output: Output): Unit = {
    output.writeInt(words.size)
    words.foreach {
      case (key, id) ⇒
        output.writeString(key)
        output.writeInt(id)
    }
  }
}

/**
 * Class for word vectors, or ledger.
 */
class LedgerModel extends Serializable with KryoSerializable {
  /** Dimension of vector */
  lazy final val dimension = vectors.head.size
  /** ID of pad */
  lazy final val padID = wordmap.padID
  /** size of model */
  lazy final val size = wordmap.size
  /** ID of Unknown */
  lazy final val unkID = wordmap.unkID
  /** list of vector */
  val vectors = mutable.ArrayBuffer[DataVec]()
  /** map of words, ledger words. */
  var wordmap = new LedgerWords()

  /**
   * Find vector of given string.
   * @param str word
   * @return Word vector
   */
  def apply(str: String): DataVec = {
    vectorAt(indexOf(str))
  }

  /**
   * Copy this ledger model
   * @return new ledger model.
   */
  def copy: LedgerModel = {
    val model = new LedgerModel().set(this.wordmap, this.vectors)
    model
  }

  /**
   * Find index of given word
   * @param str word
   * @return index
   */
  def indexOf(str: String) = wordmap.indexOf(str: String)

  /**
   * Save ledger model into text file
   * @param file Save path
   */
  def saveAsTextFile(file: Path) =
    try {
      val bos = File(file).bufferedWriter()
      wordmap.words.foreach {
        case (word, id) ⇒
          val vec = vectors(id)
          bos.write(word + " " + vec.data.mkString(" ") + "\n")
      }
      bos.close()
    } catch {
      case _: Throwable ⇒
    }

  /**
   * Save ledger model into kryo file
   * @param file Save path
   */
  def saveTo(file: Path) =
    try {
      val oos = new Output(File(file).outputStream())
      this.write(KryoWrap.kryo, oos)
      oos.close()
    } catch {
      case _: Throwable ⇒
    }

  /**
   * Set words and vectors
   * @param map word mapping, ledger words
   * @param vec sequence of vectors
   * @return self
   */
  def set(map: LedgerWords, vec: mutable.ArrayBuffer[DataVec]): this.type = {
    this.wordmap = map
    vectors ++= vec
    this
  }

  /**
   * Get vectors at index
   * @param at index
   * @return vector
   */
  def vectorAt(at: Int) = vectors(at)

  override def read(kryo: Kryo, input: Input): Unit = {
    wordmap = kryo.readClassAndObject(input).asInstanceOf[LedgerWords]
    vectors.clear()
    val size = input.readInt()
    (0 until size).foreach { _ ⇒
      val vec = kryo.readClassAndObject(input).asInstanceOf[DataVec]
      vectors += vec
    }
  }

  // Please change word layer if change this.
  override def write(kryo: Kryo, output: Output): Unit = {
    kryo.writeClassAndObject(output, wordmap)
    output.writeInt(vectors.size)
    vectors.foreach {
      case vec ⇒
        kryo.writeClassAndObject(output, vec)
    }
  }
}

/**
 * Companion class of ledger model.
 */
object LedgerModel {
  /** String for pad */
  final val PAD: String = "#PAD_X"
  /** String for unknown */
  final val UnkAll = "#SHAPE_*"
  /** Logger */
  val logger = Logger.getLogger(this.getClass)

  /**
   * Read model from path. File can be white-space splited text list.
   * @param path Path of file
   * @return LedgerModel
   */
  def read(path: Path): LedgerModel = read(File(path))

  /**
   * Read model from path. File can be white-space splited text list.
   * @param file File
   * @return LedgerModel
   */
  def read(file: File): LedgerModel = {
    val path = file.path + ".obj"
    if (File(path).exists) {
      val in = new Input(File(path).inputStream())
      val model = new LedgerModel
      model.read(KryoWrap.kryo, in)
      in.close()

      logger info s"READ Embedding Vectors finished (Dimension ${model.dimension}, Size ${model.size})"
      model
    } else {
      val br = file.lines(Codec.UTF8)

      val wordmap = new LedgerWords()
      val vectors = mutable.ArrayBuffer[DataVec]()
      val unkCounts = mutable.HashMap[String, (Int, Int)]()
      val unkvecs = mutable.ArrayBuffer[DataVec]()
      var lineNo = 0

      while (br.hasNext) {
        if (lineNo % 10000 == 0)
          logger info f"READ GloVe file (${file.name}) : $lineNo%9d"

        val line = br.next()
        val splits = line.trim().split("\\s+")
        val word = splits(0)
        val vector = DenseVector(splits.tail.map(_.toDouble))

        wordmap.words += word → lineNo
        vectors += vector
        lineNo += 1

        unkCounts.get(UnkAll) match {
          case Some((id, cnt)) ⇒
            unkvecs(id) += vector
            unkCounts(UnkAll) = (id, cnt + 1)
          case None ⇒
            unkCounts += UnkAll →(unkvecs.length, 1)
            unkvecs += vector.copy
        }

        val shape = word.getShape
        unkCounts.get(shape) match {
          case Some((id, cnt)) ⇒
            unkvecs(id) += vector
            unkCounts(shape) = (id, cnt + 1)
          case None ⇒
            unkCounts += shape →(unkvecs.length, 1)
            unkvecs += vector.copy
        }
      }

      val shapeThreshold = wordmap.size * 0.0001
      logger info s"Generate shapes for unknown vectors (${unkCounts.count(_._2._2 > shapeThreshold)} shapes)"

      unkCounts.par.map {
        case (shape, (id, count)) if count > shapeThreshold ⇒
          val matx = unkvecs(id)
          matx :/= count.toDouble
          wordmap.words += shape → vectors.length
          vectors += matx
        case _ ⇒
      }

      wordmap.words += PAD → lineNo
      vectors += vectors(wordmap.words(UnkAll)).copy

      val model = new LedgerModel().set(wordmap, vectors)
      logger info s"Start to save GloVe (Dimension ${model.dimension}, Size ${model.size})"
      model.saveTo(path)

      logger info s"READ Embedding Vectors finished (Dimension ${model.dimension}, Size ${model.size})"
      model
    }
  }
}