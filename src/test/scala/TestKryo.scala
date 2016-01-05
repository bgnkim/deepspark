import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.layer.BasicLayer
import com.github.nearbydelta.deepspark.network.{Network, SimpleNetwork}

/**
 * Created by bydelta on 15. 10. 16.
 */
object TestKryo {
  def main(args: Array[String]) {
    val builder = new AdaGrad(l2decay = 1e-6, rate = 0.01)
    val network = new SimpleNetwork[Boolean]()
      .add(new BasicLayer withInput 2 withOutput 4)
      .add(new BasicLayer withOutput 1)
      .initiateBy(builder)

    require(network.NOut == 1)

    network.saveTo("testfile")
    val net2 = Network.readFrom[SimpleNetwork[Boolean]]("testfile")

    require(net2.NOut == 1)
    require(network.layers.size == net2.layers.size)
    require(net2.layers.head.asInstanceOf[BasicLayer].bias != null)
    require(net2.layers.head.asInstanceOf[BasicLayer].weight.value != null)
    require(net2.layers.head.asInstanceOf[BasicLayer].bias.value.length > 0)
  }
}
