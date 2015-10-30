package com.github.nearbydelta.deepspark.train

import java.text.SimpleDateFormat
import java.util.Date

import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.network.Network
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.random.BernoulliSampler

import scala.concurrent.duration._
import scala.concurrent.{Await, Future}
import scala.reflect.ClassTag
import scala.reflect.io.File

/**
 * Created by bydelta on 15. 10. 29.
 */
trait Trainer[IN, EXP, OUT] extends Serializable {
  var network: Network[IN, OUT]
  val objective: Objective
  val name: String
  val param: TrainingParam

  implicit val evidence$1: ClassTag[IN]
  implicit val evidence$2: ClassTag[OUT]
  implicit val evidence$3: ClassTag[EXP]

  @transient protected val trainSet: RDD[(IN, EXP)]
  @transient protected val testSet: RDD[(IN, EXP)]
  @transient protected val sampleFraction: Double = param.miniBatch.toDouble / trainSet.count()
  /** Period of validation */
  @transient protected val validationPeriod: Int = (param.validationFreq / sampleFraction).toInt
  @transient protected val file = File(name + ".kryo")
  @transient protected val logger = Logger.getLogger(this.getClass)
  @transient private val startAt = System.currentTimeMillis()

  /** Best Loss Iteration Number */
  @transient protected var bestIter: Int = 0
  @transient protected var prevLossW: Double = Double.PositiveInfinity
  @transient protected var prevLossE: Double = Double.PositiveInfinity

  @transient private final val dateFormatter = new SimpleDateFormat("yy/MM/dd HH:mm:ss")
  @transient private final val columns = try {
    System.getenv("COLUMNS").toInt
  } catch {
    case _: Throwable ⇒ 80
  }

  class RDDSampler extends Iterator[Seq[(IN, EXP)]] {
    val sampleSize = (100000 / param.miniBatch) * sampleFraction
    var temp = trainSet.sample(withReplacement = true, fraction = sampleSize)
      .collect().toIterator.sliding(param.miniBatch, param.miniBatch)
    var tempF: Future[Seq[(IN, EXP)]] = attachNext()

    def attachNext() =
      trainSet.sample(withReplacement = true, fraction = sampleSize).collectAsync()

    override def hasNext: Boolean = true

    override def next(): Seq[(IN, EXP)] = {
      val next = temp.next()
      if (!temp.hasNext) {
        temp = Await.result(tempF, 1.hour).toIterator.sliding(param.miniBatch, param.miniBatch)
        tempF = attachNext()
      }
      next
    }
  }

  class LocalSampler extends Iterator[Seq[(IN, EXP)]] {
    val localData = Await.result(trainSet.collectAsync(), 1.hour).toIterator
    val sampler = new BernoulliSampler[(IN, EXP)](sampleFraction)

    override def hasNext: Boolean = true

    override def next(): Seq[(IN, EXP)] = {
      sampler.sample(localData).toSeq.take(param.miniBatch)
    }
  }

  @transient protected val batchSampler =
    if (param.dataOnLocal) new LocalSampler
    else new RDDSampler


  private final def printProgress(iter: Int, patience: Int, lossE: Double, lossW: Double): Unit = {
    val wait = patience / param.maxIter.toFloat
    val header = f"\033[4m$name\033[24m $iter%4d/$patience%4d \033[0m["
    val footer = f" E + W = $lossE%7.5f + $lossW%7.5f "
    val secondFooter = f" vs $prevLossE%7.5f + $prevLossW%7.5f @ $bestIter%4d "

    val buf = new StringBuilder(s"\033[2A\033[${columns}D\033[2K \033[1;33m$header\033[46;36m")
    val total = columns - header.length - footer.length + 10
    val len = Math.floor(wait * total).toInt
    val step = Math.floor(iter / param.maxIter.toFloat * total).toInt
    buf.append(" " * step)
    buf.append("\033[49m")
    buf.append(" " * (len - step))
    buf.append("\033[0m]\033[34m")
    if (total > len) buf.append(s"\033[${total - len}C")
    buf.append(s"$footer\033[0m\n\033[2K")

    val now = System.currentTimeMillis()
    val remainA = (now - startAt) / iter * patience
    val etaA = startAt + remainA
    val calA = dateFormatter.format(new Date(etaA))
    val remainB = (now - startAt) / iter * param.maxIter
    val etaB = startAt + remainB
    val calB = dateFormatter.format(new Date(etaB))
    val timeline = s" END $calA   ~   $calB"
    buf.append(timeline)
    buf.append(" " * (columns - timeline.length - secondFooter.length))
    buf.append(secondFooter)

    println(buf.result())
  }

  /**
   * Train using given RDD sequence.
   */
  final def getTrainedNetwork: Network[IN, OUT] =
    if (validationPeriod > 0) {
      val rootLogger: Logger = Logger.getLogger("org")
      // Turnoff spark logging feature.
      val lv = rootLogger.getLevel
      rootLogger.setLevel(Level.WARN)

      logger info f"($name) Starts training. "
      logger info f"($name) Every $validationPeriod%5d Epoch (${param.validationFreq * 100}%6.2f%% of TrainingSet; " +
        f"${param.miniBatch * validationPeriod}%6d Instances), validation process will be submitted."

      if (file.exists && param.reuseSaveData) {
        loadStatus()
        logger info f"We detected a network saved at $bestIter%4d Iteration." +
          f"with loss $prevLossE%8.5f + $prevLossW%8.5f. We use it as initial value. Loss will be reset."
        prevLossE = Double.PositiveInfinity
      }else
        saveStatus()
      network.setUpdatable(true)
      println("Start training...\n Estimated Time: NONE")

      val epoch = bestIter * validationPeriod
      val patience = Math.min(Math.max(5, bestIter) * (param.waitAfterUpdate + 1), param.maxIter)
      trainSmallBatch(epoch, patience)
      loadStatus()
      network.setUpdatable(false)

      rootLogger.setLevel(lv)
      network
    } else {
      logger warn f"($name) Validation Period is zero! Training stopped."
      logger warn f"($name) Maybe because miniBatchFraction value is too large. Please check."
      network
    }

  private final def saveStatus() = {
    val output = new Output(file.outputStream())
    val kryo = KryoWrap.kryo
    output.writeInt(bestIter)
    output.writeDouble(prevLossE)
    output.writeDouble(prevLossW)
    kryo.writeClassAndObject(output, network)
    output.close()
  }

  private final def loadStatus() = {
    val input = new Input(file.inputStream())
    val kryo = KryoWrap.kryo
    bestIter = input.readInt()
    prevLossE = input.readDouble()
    prevLossW = input.readDouble()
    network = kryo.readClassAndObject(input).asInstanceOf[Network[IN, OUT]]
    input.close()
  }

  protected def getValidationErr: Double

  protected def trainSmallBatch(epoch: Int, patience: Int): Unit

  protected final def getMaxEpoch(epoch: Int, patience: Int): Int = {
    val iter = epoch / validationPeriod + 1
    var nPatience = patience
    var prevloss = prevLossW + prevLossE

    if ((epoch + 1) % validationPeriod == 0) {
      val train = getValidationErr
      val weight = network.loss
      val loss = train + weight
      if (loss.isNaN || loss.isInfinity) {
        logger info s"Because of some issue, loss became $loss (train $train, weight $weight). Please check."
        logger info s"Automatically reload latest successful setup.\n\n"
        loadStatus()
      } else {
        val improvement = if (prevloss > 0f) loss / prevloss else param.improveThreshold
        if (improvement < param.improveThreshold) {
          nPatience = Math.min(Math.max(patience, iter * (param.waitAfterUpdate + 1)), param.maxIter)

          bestIter = iter
          prevLossE = train
          prevLossW = weight
          prevloss = loss
          saveStatus()
        }
      }
      printProgress(iter, nPatience, train, weight)
    }

    if (iter <= nPatience && (prevloss >= param.lossThreshold || iter < 5)) {
      nPatience
    } else {
      if (prevloss < param.lossThreshold)
        logger info f"($name) # $iter%4d/$nPatience%4d, " +
          f"FINISHED with E + W = $prevloss%.5f [Loss < ${param.lossThreshold}%.5f]"
      else if (iter > param.maxIter)
        logger info f"($name) # $iter%4d/$nPatience%4d, " +
          f"FINISHED with E + W = $prevloss%.5f [Iteration > ${param.maxIter}%6d]"
      else if (nPatience < iter)
        logger info f"($name) # $iter%4d/$nPatience%4d, " +
          f"FINISHED with E + W = $prevloss%.5f [NoUpdate after $bestIter%6d]"
      0
    }
  }
}