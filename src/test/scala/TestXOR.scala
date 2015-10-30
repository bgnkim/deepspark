import breeze.linalg.DenseVector
import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.layer.BasicLayer
import com.github.nearbydelta.deepspark.network.SimpleNetwork
import com.github.nearbydelta.deepspark.train.{TrainerBuilder, TrainingParam}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by bydelta on 15. 10. 16.
 */
object TestXOR {
  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local[5]").setAppName("TestXOR")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.broadcast.blockSize", "40960")
      .set("spark.akka.frameSize", "50")
    val sc = new SparkContext(conf)

    val data = (0 to 100).collect{
      case i if i > 75 || i < 25 ⇒
      (0 to 100).collect{
        case j if j > 75 || j < 25 ⇒
        val xor =
          if (i > 75 && j > 75) true
          else if (i < 25 && j < 25) true
          else false
        (DenseVector[Double](i / 100.0, j / 100.0), xor)
      }
    }.flatMap(x ⇒ x)

    val train = sc.makeRDD(data)
    val test = train

    try {
      val builder = new AdaGrad(l2decay = 0.0, rate = 0.6)
      val network = new SimpleNetwork[Boolean]()
        .initiateBy(builder)
        .add(new BasicLayer withInput 2 withOutput 4)
        .add(new BasicLayer withOutput 1)

      require(network.NOut == 1)
      require(network.layers.head.asInstanceOf[BasicLayer].bias != null)
      require(network.layers.head.asInstanceOf[BasicLayer].weight.value != null)
      require(network.layers.head.asInstanceOf[BasicLayer].bias.value.length > 0)

      val trained = new TrainerBuilder(TrainingParam(miniBatch = 10, maxIter = 5000, dataOnLocal = true,
        reuseSaveData = true, storageLevel = StorageLevel.MEMORY_ONLY))
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
