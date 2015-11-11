package com.github.nearbydelta.deepspark.train

import com.github.nearbydelta.deepspark.data.{DataVec, Objective}
import com.github.nearbydelta.deepspark.network.Network
import org.apache.spark.rdd.RDD

import scala.annotation.tailrec
import scala.reflect.ClassTag


/**
 * __Builder__ for trainer.
 *
 * @param param Training parameter
 */
class TrainerBuilder(val param: TrainingParam = TrainingParam()) {

  /**
   * Train network
   * @param network Network
   * @param trainset Train set
   * @param testset Evaluation set
   * @param objective Objective function
   * @param expectConverter Function to convert real output into vector
   * @param mapper Function to convert item into (in, out) pair
   * @param name Name of trainer.
   * @param evidence$1 class tag of input (implicit)
   * @param evidence$2 class tag of output (implicit)
   * @tparam X Type of given set
   * @tparam IN Type of input
   * @tparam OUT Type of real output
   * @return trained network.
   */
  def train[X, IN, OUT](network: Network[IN, OUT],
                        trainset: RDD[X],
                        testset: RDD[X],
                        objective: Objective,
                        expectConverter: OUT ⇒ DataVec,
                        mapper: X ⇒ (IN, OUT),
                        name: String = "temp")
                       (implicit evidence$1: ClassTag[IN], evidence$2: ClassTag[OUT]): Network[IN, OUT] =
    train(network, trainset.map(mapper), testset.map(mapper), objective, expectConverter, name)

  /**
   * Train network
   * @param network Network
   * @param trainset Train set
   * @param testset Evaluation set
   * @param objective Objective function
   * @param expectConverter Function to convert real output into vector
   * @param name Name of trainer.
   * @param evidence$1 class tag of input (implicit)
   * @param evidence$2 class tag of output (implicit)
   * @tparam IN Type of input
   * @tparam OUT Type of real output
   * @return trained network.
   */
  def train[IN, OUT](network: Network[IN, OUT],
                     trainset: RDD[(IN, OUT)],
                     testset: RDD[(IN, OUT)],
                     objective: Objective,
                     expectConverter: OUT ⇒ DataVec,
                     name: String)
                    (implicit evidence$1: ClassTag[IN], evidence$2: ClassTag[OUT]): Network[IN, OUT] = {
    val train = trainset.mapValues(expectConverter).setName(name + " (TRAIN)").persist(param.storageLevel)
    val test = testset.mapValues(expectConverter).setName(name + " (EVAL)").persist(param.storageLevel)
    new StaticTrainer[IN, OUT](network, objective, train, test, name, param).getTrainedNetwork
  }

  /**
   * Train network
   * @param network Network
   * @param trainset Train set
   * @param testset Evaluation set
   * @param objective Objective function
   * @param name Name of trainer.
   * @param evidence$1 class tag of input (implicit)
   * @tparam IN Type of input
   * @return trained network.
   */
  def train[IN](network: Network[IN, DataVec],
                trainset: RDD[(IN, DataVec)],
                testset: RDD[(IN, DataVec)],
                objective: Objective,
                name: String)
               (implicit evidence$1: ClassTag[IN]): Network[IN, DataVec] = {
    val train = trainset.setName(name + " (TRAIN)").persist(param.storageLevel)
    val test = testset.setName(name + " (EVAL)").persist(param.storageLevel)
    new StaticTrainer[IN, DataVec](network, objective, train, test, name, param).getTrainedNetwork
  }

  /**
   * Train network, real output vector is not fixed before evaluation.
   * @param network Network
   * @param trainset Train set
   * @param testset Evaluation set
   * @param objective Objective function
   * @param expectConverter Function to convert real output into vector
   * @param name Name of trainer.
   * @param evidence$1 class tag of input (implicit)
   * @param evidence$2 class tag of output (implicit)
   * @tparam IN Type of input
   * @tparam OUT Type of real output
   * @return trained network.
   */
  def trainFlexible[IN, OUT](network: Network[IN, OUT],
                             trainset: RDD[(IN, OUT)],
                             testset: RDD[(IN, OUT)],
                             objective: Objective,
                             expectConverter: OUT ⇒ DataVec,
                             name: String)
                            (implicit evidence$1: ClassTag[IN], evidence$2: ClassTag[OUT]): Network[IN, OUT] = {
    val train = trainset.setName(name + " (TRAIN)").persist(param.storageLevel)
    val test = testset.setName(name + " (EVAL)").persist(param.storageLevel)
    new FlexibleTrainer[IN, OUT](network, objective, expectConverter, train, test, name, param).getTrainedNetwork
  }
}

/**
 * Static real-output trainer. (Fix real output vector before evaluation)
 * @param network Network
 * @param objective Objective function
 * @param trainSet Train set
 * @param testSet Evaluation set
 * @param name Name of trainer
 * @param param Training parameter
 * @param evidence$1 class tag of input
 * @param evidence$2 class tag of output
 * @param evidence$3 class tag of vector
 * @tparam IN Input type
 * @tparam OUT Output type
 */
private class StaticTrainer[IN, OUT](var network: Network[IN, OUT],
                                     val objective: Objective,
                                     val trainSet: RDD[(IN, DataVec)],
                                     val testSet: RDD[(IN, DataVec)],
                                     val name: String,
                                     val param: TrainingParam)
                                    (implicit override val evidence$1: ClassTag[IN],
                                     implicit override val evidence$2: ClassTag[OUT],
                                     implicit override val evidence$3: ClassTag[DataVec])
  extends Trainer[IN, DataVec, OUT] {

  @tailrec
  override protected final def trainSmallBatch(epoch: Int, patience: Int): Unit = {
    val trainSample = batchSampler.next()

    val input = trainSample.map(_._1).par
    val output = network.forward(input)

    val error = trainSample.map(_._2).par.zip(output).map {
      case (exp, out) ⇒
        val err = objective.derivative(exp, out)
        require(err.data.count(x ⇒ x.isNaN || x.isInfinity) == 0,
          s"Please check your objective function. Derivative has NaN/Infinity! (Expected: $exp, Network: $out)")
        err
    }

    network.backward(error)

    val maxEpoch = getMaxEpoch(epoch, patience)
    if (maxEpoch > 0)
      trainSmallBatch(epoch + 1, maxEpoch)
  }

  override protected def getValidationErr: (Double, Double) = {
    network.broadcast(trainSet.context)
    val loss = testSet.context.accumulator(0.0)
    val count = testSet.context.accumulator(0)
    val nanCount = testSet.context.accumulator(0)
    testSet.foreachPartition {
      it ⇒
        var sum = 0.0
        var cnt = 0

        it.toStream.par.foreach {
          case (i, o) ⇒
            val output = network.forward(i)
            val obj: Double = objective(o, output)

            if (obj.isNaN || obj.isInfinite)
              nanCount += 1
            else {
              sum += obj
              cnt += 1
            }
        }

        loss += sum
        count += cnt
    }
    network.unbroadcast()

    if (nanCount.value > 0)
      logger warn s"There was ${nanCount.value} NaN/Infinity value during Evaluation Process!"

    val weight = network.loss

    if (count.value == 0) {
      logger warn "There's no regular instance inside evaluation set!"
      (Double.PositiveInfinity, Double.PositiveInfinity)
    } else {
      (loss.value / count.value, weight / count.value)
    }
  }
}

/**
 * Flexible real-output trainer. (Real output vector re-calcuated every evaluation.)
 * @param network Network
 * @param objective Objective function
 * @param trainSet Train set
 * @param testSet Evaluation set
 * @param name Name of trainer
 * @param param Training parameter
 * @param evidence$1 class tag of input
 * @param evidence$2 class tag of output
 * @tparam IN Input type
 * @tparam OUT Output type
 */
private class FlexibleTrainer[IN, OUT](var network: Network[IN, OUT],
                                       val objective: Objective,
                                       val expectConverter: OUT ⇒ DataVec,
                                       val trainSet: RDD[(IN, OUT)],
                                       val testSet: RDD[(IN, OUT)],
                                       val name: String,
                                       val param: TrainingParam)
                                      (implicit override val evidence$1: ClassTag[IN],
                                       implicit override val evidence$2: ClassTag[OUT])
  extends Trainer[IN, OUT, OUT] {
  override implicit val evidence$3: ClassTag[OUT] = evidence$2

  @tailrec
  override protected final def trainSmallBatch(epoch: Int, patience: Int): Unit = {
    val trainSample = batchSampler.next()

    val input = trainSample.map(_._1).par
    val output = network.forward(input)

    val error = trainSample.map(_._2).par.zip(output).map {
      case (exp, out) ⇒
        val err = objective.derivative(expectConverter(exp), out)
        require(err.data.count(x ⇒ x.isNaN || x.isInfinity) == 0,
          s"Please check your objective function. Derivative has NaN/Infinity! (Expected: $exp, Network: $out)")
        err
    }

    network.backward(error)

    val maxEpoch = getMaxEpoch(epoch, patience)
    if (maxEpoch > 0)
      trainSmallBatch(epoch + 1, maxEpoch)
  }

  override protected def getValidationErr: (Double, Double) = {
    network.broadcast(trainSet.context)
    val loss = testSet.context.accumulator(0.0)
    val count = testSet.context.accumulator(0)
    val nanCount = testSet.context.accumulator(0)
    testSet.foreachPartition {
      it ⇒
        var sum = 0.0
        var cnt = 0

        it.toStream.par.foreach {
          case (i, o) ⇒
            val output = network.forward(i)
            val obj: Double = objective(expectConverter(o), output)

            if (obj.isNaN || obj.isInfinite)
              nanCount += 1
            else {
              sum += obj
              cnt += 1
            }
        }

        loss += sum
        count += cnt
    }
    network.unbroadcast()

    if (nanCount.value > 0)
      logger warn s"There was ${nanCount.value} NaN/Infinity value during Evaluation Process!"

    val weight = network.loss

    if (count.value == 0) {
      logger warn "There's no regular instance inside evaluation set!"
      (Double.PositiveInfinity, Double.PositiveInfinity)
    } else {
      (loss.value / count.value, weight / count.value)
    }
  }
}
