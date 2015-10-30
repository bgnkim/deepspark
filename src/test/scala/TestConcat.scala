import breeze.linalg.DenseVector
import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.layer.{BasicLayer, NetworkConcatLayer}
import com.github.nearbydelta.deepspark.network.{GeneralNetwork, SimpleNetwork}
import com.github.nearbydelta.deepspark.train.{TrainerBuilder, TrainingParam}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

import scala.reflect.{ClassTag, classTag}

/**
 * Created by bydelta on 15. 10. 16.
 */
object TestConcat {
  class ConcatLayer extends NetworkConcatLayer[DataVec]{
    override implicit protected val evidenceI: ClassTag[Array[DataVec]] = classTag[Array[DataVec]]
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local[5]").setAppName("TestXOR")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.broadcast.blockSize", "40960")
      .set("spark.akka.frameSize", "50")
    val sc = new SparkContext(conf)

    val data = (0 to 10).collect{
      case i if i > 7 || i < 3 ⇒
        (0 to 10).collect{
          case j if j > 7 || j < 3 ⇒
            val xor =
              if (i > 7 && j > 7) true
              else if (i < 3 && j < 3) true
              else false
            (0 to 10).collect{
              case k if k > 7 || k < 3 ⇒
                (0 to 10).collect{
                  case l if l > 7 || l < 3 ⇒
                    val xor2 =
                      if (i > 7 && j > 7) true
                      else if (i < 3 && j < 3) true
                      else false
                    (Array(DenseVector(i / 10.0, j / 10.0), DenseVector(k / 10.0, l / 10.0)),
                      xor && xor2)
                }
            }.flatMap(x ⇒ x)
        }.flatMap(x ⇒ x)
    }.flatMap(x ⇒ x)

    val train = sc.makeRDD(data)
    val test = train

    try {
      val builder = new AdaGrad(l2decay = 0.001, rate = 0.1)
      val input1 = new SimpleNetwork[DataVec]()
        .initiateBy(builder)
        .add(new BasicLayer withInput 2 withOutput 4)
        .add(new BasicLayer withInput 4 withOutput 1)
      val input2 = new SimpleNetwork[DataVec]()
        .initiateBy(builder)
        .add(new BasicLayer withInput 2 withOutput 4)
        .add(new BasicLayer withInput 4 withOutput 1)
      val concat = new ConcatLayer().addNetwork(input1).addNetwork(input2)
      val network = new GeneralNetwork[Array[DataVec], Boolean](concat)
        .initiateBy(builder)
        .add(new BasicLayer withInput 2 withOutput 4)
        .add(new BasicLayer withInput 4 withOutput 1)

      require(network.NOut == 1)

      val trained = new TrainerBuilder(TrainingParam(miniBatch = 100, maxIter = 100, storageLevel = StorageLevel.MEMORY_ONLY))
        .train(network, train, test, SquaredErr, (x: Boolean) ⇒ if (x) DenseVector(1.0) else DenseVector(0.0), "XORTest")

      (0 until 10).foreach { _ ⇒
        val (in, exp) = data(Math.floor(Math.random() * data.length).toInt)
        val out = trained.predictSoft(in)
        println(s"IN : $in, EXPECTED: $exp, OUTPUT $out")
      }
    }finally {
      sc.stop()
    }
  }
}
