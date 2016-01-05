import breeze.linalg.{DenseVector, sum}
import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.layer.BasicLayer
import com.github.nearbydelta.deepspark.network.{GeneralNetwork, Network}
import com.github.nearbydelta.deepspark.train.{TrainerBuilder, TrainingParam}
import com.github.nearbydelta.deepspark.word.layer.RNNLedger
import com.github.nearbydelta.deepspark.word.{LedgerAdaGrad, LedgerModel, LedgerWords}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable

/**
 * Created by bydelta on 15. 10. 16.
 */
object TestEmbedding {
  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local[5]").setAppName("TestXOR")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.broadcast.blockSize", "40960")
      .set("spark.akka.frameSize", "50")
    val sc = new SparkContext(conf)

    val letters = new LedgerWords()
    val vectors = mutable.ArrayBuffer[DataVec]()
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ".sliding(1).foreach {
      letter ⇒
        letters.words += letter → letters.words.size
        vectors += DenseVector.rand(5)
    }

    /* Testing example: learn the nearest vowel letter */
    @transient val example = Seq("A" → DenseVector(1.0, 0.0, 0.0, 0.0, 0.0),
      "B" → DenseVector(.75, .25, 0.0, 0.0, 0.0),
      "C" → DenseVector(0.5, 0.5, 0.0, 0.0, 0.0),
      "D" → DenseVector(.25, .75, 0.0, 0.0, 0.0),
      "E" → DenseVector(0.0, 1.0, 0.0, 0.0, 0.0),
      "F" → DenseVector(0.0, .75, .25, 0.0, 0.0),
      "G" → DenseVector(0.0, 0.5, 0.5, 0.0, 0.0),
      "H" → DenseVector(0.0, .25, .75, 0.0, 0.0),
      "I" → DenseVector(0.0, 0.0, 1.0, 0.0, 0.0),
      "J" → DenseVector(0.0, 0.0, 0.8, 0.2, 0.0),
      "K" → DenseVector(0.0, 0.0, .67, .33, 0.0),
      "L" → DenseVector(0.0, 0.0, 0.5, 0.5, 0.0),
      "M" → DenseVector(0.0, 0.0, .33, .67, 0.0),
      "N" → DenseVector(0.0, 0.0, 0.2, 0.8, 0.0),
      "O" → DenseVector(0.0, 0.0, 0.0, 1.0, 0.0),
      "P" → DenseVector(0.0, 0.0, 0.0, 0.8, 0.2),
      "Q" → DenseVector(0.0, 0.0, 0.0, .67, .33),
      "R" → DenseVector(0.0, 0.0, 0.0, 0.5, 0.5),
      "S" → DenseVector(0.0, 0.0, 0.0, .33, .67),
      "T" → DenseVector(0.0, 0.0, 0.0, 0.2, 0.8),
      "U" → DenseVector(0.0, 0.0, 0.0, 0.0, 1.0),
      "V" → DenseVector(0.0, 0.0, 0.0, 0.0, 1.0),
      "W" → DenseVector(0.0, 0.0, 0.0, 0.0, 1.0),
      "X" → DenseVector(0.0, 0.0, 0.0, 0.0, 1.0),
      "Y" → DenseVector(0.0, 0.0, 0.0, 0.0, 1.0),
      "Z" → DenseVector(0.0, 0.0, 0.0, 0.0, 1.0)).combinations(3).map {
      it ⇒
        val word = it.map(x ⇒ letters.indexOf(x._1)).toArray
        val vec = it.map(_._2).reduceLeft[DataVec] {
          case (left, right) ⇒
            val l: DataVec = left * 0.8
            val r: DataVec = right * .2
            l + r
        }
        word → vec
    }.toSeq

    val vectorA = vectors(letters.indexOf("A")).copy

    val data = sc.parallelize(example)

    try {
      val wb = new AdaGrad(l2decay = 0.001, rate = 0.01)
      val mb = new LedgerAdaGrad(l2decay = 0.0, rate = 0.01)

      val embedding = new LedgerModel
      embedding.set(letters, vectors)


      val wordlayer = new BasicLayer withInput 10 withOutput 5
      val network = new GeneralNetwork[Array[Int], DataVec](new RNNLedger(wordlayer).withModel(embedding.copy, mb))
        .add(new BasicLayer withInput 5 withOutput 10)
        .add(new BasicLayer withActivation SoftmaxCEE withInput 10 withOutput 5)
        .initiateBy(wb)

      val trained = new TrainerBuilder(TrainingParam(miniBatch = 10, maxIter = 1000,
        reuseSaveData = true, storageLevel = StorageLevel.MEMORY_ONLY))
        .build(network, data, data, SquaredErr, "EmbeddingTest")
        .getTrainedNetwork

      trained.saveTo("testfile")
      val net2 = Network.readFrom[GeneralNetwork[Array[Int], DataVec]]("testfile")
      val newA = net2.inputLayer.asInstanceOf[RNNLedger].model.apply("A")

      (0 until 10).foreach { _ ⇒
        val (in, exp) = example(Math.floor(Math.random() * example.length).toInt)
        val out = trained.predictSoft(in)
        println(s"IN : $in, EXPECTED: $exp, OUTPUT $out")
        val out2 = net2.predictSoft(in)
        require(sum(out - out2) == 0.0)
      }

      println(newA)
      println(vectorA)
      require(sum(newA - vectorA) != 0.0)
    } finally {
      sc.stop()
    }
  }
}
