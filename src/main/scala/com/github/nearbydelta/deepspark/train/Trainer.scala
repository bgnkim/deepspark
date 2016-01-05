package com.github.nearbydelta.deepspark.train

import java.io._
import java.text.SimpleDateFormat
import java.util.Date

import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.nearbydelta.deepspark.data._
import com.github.nearbydelta.deepspark.network.Network
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.SizeEstimator

import scala.concurrent._
import scala.concurrent.duration._
import scala.reflect.ClassTag
import scala.util.Random

/**
 * __Trait__ of trainer
 * @tparam IN Input type
 * @tparam EXP Network output type
 * @tparam OUT Output type
 */
trait Trainer[IN, EXP, OUT] extends Serializable {
  /** Width of command line */
  @transient private final val columns = try {
    System.getenv("COLUMNS").toInt
  } catch {
    case _: Throwable ⇒ 80
  }
  /** Date formatter */
  @transient private final val dateFormatter = new SimpleDateFormat("yy/MM/dd HH:mm:ss")
  /** Objective function */
  val objective: Objective
  /** Name of this trainer */
  val name: String
  /** Training parameter */
  val param: TrainingParam
  /** Class tag of input */
  implicit val evidence$1: ClassTag[IN]
  /** Class tag of output */
  implicit val evidence$2: ClassTag[OUT]
  /** Class tag of network output */
  implicit val evidence$3: ClassTag[EXP]
  /** Sampler for mini-batch */
  @transient protected val batchSampler =
    if (param.dataOnLocal) new LocalSampler
    else new RDDSampler
  /** Temporary file path of network */
  @transient protected val file = new File(name + ".kryo")
  @transient protected val logfile = new File(name + ".csv")
  /** Logger */
  @transient protected val logger = Logger.getLogger(this.getClass)
  /** Train set */
  @transient protected val trainSet: RDD[(IN, EXP)]
  /** Evaluation set */
  @transient protected val testSet: RDD[(IN, EXP)]
  /** Time of begin */
  @transient private val startAt = System.currentTimeMillis()
  /** Sampling fraction per minibatch */
  @transient private val trainSize: Long = trainSet.count()
  /** Period of validation */
  @transient private val validationPeriod: Int = (param.validationFreq * trainSize / param.miniBatch).toInt
  /** Network */
  var network: Network[IN, OUT]
  /** Best Loss Iteration Number */
  @transient private var bestIter: Int = 0
  /** Evaluation loss of best iter */
  @transient private var prevLossE: Double = Double.PositiveInfinity
  /** Weight loss of best iter */
  @transient private var prevLossW: Double = Double.PositiveInfinity

  def getStatus = (bestIter, prevLossE, prevLossW)

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
          f"with loss $prevLossE%8.5f + $prevLossW%8.5f. We use it as initial value."
      } else {
        val (e, w) = getValidationErr
        prevLossE = e
        prevLossW = w
        saveStatus()
      }
      print("\n\n")
      network.setUpdatable(true)

      val epoch = bestIter * validationPeriod
      val patience = Math.min(Math.max(5, bestIter * (param.waitAfterUpdate + 1)), param.maxIter)
      printProgress(bestIter, patience, prevLossE, prevLossW)
      trainSmallBatch(epoch, patience)
      loadStatus()
      network.setUpdatable(false)
      if (file.exists())
        file.delete()

      rootLogger.setLevel(lv)
      network
    } else {
      logger warn f"($name) Validation Period is zero! Training stopped."
      logger warn f"($name) Maybe because miniBatchFraction value is too large. Please check."
      network
    }

  /**
   * Get next maximum epoch.
   * @param epoch Current epoch
   * @param patience Current maximum iteration count
   * @return maximum epoch
   */
  protected final def getMaxEpoch(epoch: Int, patience: Int): Int = {
    val iter = epoch / validationPeriod + 1
    var nPatience = patience
    var prevloss = prevLossW + prevLossE

    if ((epoch + 1) % validationPeriod == 0) {
      val (train, weight) = getValidationErr
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

    if (epoch % 10 == 0) {
      val remainder = epoch % validationPeriod
      print(f"\033[${columns}D $remainder%10d of $validationPeriod%10d epoch ...")
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

  /**
   * Calculate validation error of evaluation set
   * @return Evaluation loss
   */
  protected def getValidationErr: (Double, Double)

  /**
   * Train single small batch
   * @param epoch current epoch
   * @param patience current max iteration
   */
  protected def trainSmallBatch(epoch: Int, patience: Int): Unit

  /**
   * Load saved status
   */
  private final def loadStatus() = {
    val input = new Input(new FileInputStream(file))
    val kryo = KryoWrap.get.kryo
    bestIter = input.readInt()
    prevLossE = input.readDouble()
    prevLossW = input.readDouble()
    network = kryo.readClassAndObject(input).asInstanceOf[Network[IN, OUT]]
    input.close()
  }

  /**
   * Print current progress
   * @param iter Current iteration
   * @param patience Current maximum iteration
   * @param lossE Current evaluation loss
   * @param lossW Current weight loss
   */
  private final def printProgress(iter: Int, patience: Int, lossE: Double, lossW: Double): Unit = {
    val bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(logfile)))
    bw.append(s"$iter,$lossE,$lossW\n")
    bw.close()

    val wait = patience / param.maxIter.toFloat
    val header = f"\033[4m$name\033[24m $iter%4d/$patience%4d \033[0m["
    val footer = f" E + W = $lossE%7.5f + $lossW%7.5f "
    val secondFooter = f"[@$bestIter%4d] $prevLossE%7.5f + $prevLossW%7.5f "

    val buf = new StringBuilder(s"\033[3A\033[${columns}D\033[2K \033[1;33m$header\033[46;36m")
    val total = columns - header.length - footer.length + 10
    val len = Math.floor(wait * total).toInt
    val step = Math.floor(iter / param.maxIter.toFloat * total).toInt
    buf.append(" " * step)
    buf.append("\033[49m")
    buf.append(" " * (len - step))
    buf.append("\033[0m]\033[34m")
    if (total > len) buf.append(s"\033[${total - len}C")
    buf.append(s"$footer\033[0m\n\033[2K")

    val timeline =
      if (iter > 0) {
        val now = System.currentTimeMillis()
        val remainA = (now - startAt) / iter * patience
        val etaA = startAt + remainA
        val calA = dateFormatter.format(new Date(etaA))
        val remainB = (now - startAt) / iter * param.maxIter
        val etaB = startAt + remainB
        val calB = dateFormatter.format(new Date(etaB))
        s" END $calA   ~   $calB"
      } else {
        " END: CALCULATING..."
      }
    buf.append(timeline)
    buf.append(" " * (columns - timeline.length - secondFooter.length - 1))
    buf.append("\033[34m")
    buf.append(secondFooter)
    buf.append("\033[0m\n")

    println(buf.result())
  }

  /**
   * Save current status
   * @param path Save path
   */
  private final def saveStatus(path: File = file) = {
    val output = new Output(new FileOutputStream(path))
    val kryo = KryoWrap.get.kryo
    output.writeInt(bestIter)
    output.writeDouble(prevLossE)
    output.writeDouble(prevLossW)
    kryo.writeClassAndObject(output, network)
    output.close()
  }

  /**
   * Sampling iterator, from RDD
   */
  class RDDSampler extends Iterator[Seq[(IN, EXP)]] {
    /** Size of sample */
    val sampleSize = {
      val logger = Logger.getLogger(this.getClass)
      val sample = trainSet.takeSample(withReplacement = false, num = 10)
      val sizeByte = SizeEstimator.estimate(sample) / 10
      logger info s"Estimated training instance's size is $sizeByte byte(s)."

      val available = trainSet.context.getConf.get("spark.driver.maxResultSize", "1g").toLowerCase match {
        case x if x.endsWith("g") ⇒ x.substring(0, x.length - 1).toInt * 1024
        case x if x.endsWith("m") ⇒ x.substring(0, x.length - 1).toInt
      }

      val sizeCnt = trainSet.count()
      val parts = Math.ceil(sizeByte * sizeCnt / (1024 * 1024.0) / (available * 0.8)).toInt
      logger info s"Training set will be splited into $parts partitions, because available maxResultSize is $available MB."

      Array.fill(parts)(1.0)
    }
    /** Randomly splitted RDDs **/
    var rddSamples = trainSet.randomSplit(sampleSize).toIterator
    /** Current sampling sequence */
    var temp = Random.shuffle(rddSamples.next().collect().toSeq).sliding(param.miniBatch, param.miniBatch)

    override def hasNext: Boolean = true

    override def next(): Seq[(IN, EXP)] = {
      val next = temp.next()
      if (!temp.hasNext) {
        if (!rddSamples.hasNext)
          rddSamples = trainSet.randomSplit(sampleSize).toIterator
        temp = Random.shuffle(rddSamples.next().collect().toSeq).sliding(param.miniBatch, param.miniBatch)
      }
      next
    }
  }

  /**
   * Sampling iterator, from local
   */
  class LocalSampler extends Iterator[Seq[(IN, EXP)]] {
    /** All data of training set */
    val localData = Await.result(trainSet.collectAsync(), 1.hour)
    /** Current sampling sequence */
    var temp = Random.shuffle(localData).sliding(param.miniBatch, param.miniBatch)

    override def hasNext: Boolean = true

    override def next(): Seq[(IN, EXP)] = {
      val next = temp.next()
      if (!temp.hasNext) {
        temp = Random.shuffle(localData).sliding(param.miniBatch, param.miniBatch)
      }
      next
    }
  }

}