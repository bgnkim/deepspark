package com.github.nearbydelta.deepspark.train

import com.github.nearbydelta.deepspark.data.{DataVec, Objective}
import com.github.nearbydelta.deepspark.network.Network
import org.apache.spark.rdd.RDD

import scala.annotation.tailrec
import scala.reflect.ClassTag


/**
 * __General__ Trainer Implementation.
 */
class TrainerBuilder(val param: TrainingParam = TrainingParam()) {

  def train[X, IN, OUT](network: Network[IN, OUT],
                        trainset: RDD[X],
                        testset: RDD[X],
                        objective: Objective,
                        expectConverter: OUT ⇒ DataVec,
                        mapper: X ⇒ (IN, OUT),
                        name: String = "temp")
                       (implicit evidence$1: ClassTag[IN], evidence$2: ClassTag[OUT]): Network[IN, OUT] =
    train(network, trainset.map(mapper), testset.map(mapper), objective, expectConverter, name)

  def train[IN, OUT](network: Network[IN, OUT],
                     trainset: RDD[(IN, OUT)],
                     testset: RDD[(IN, OUT)],
                     objective: Objective,
                     expectConverter: OUT ⇒ DataVec,
                     name: String = "temp")
                    (implicit evidence$1: ClassTag[IN], evidence$2: ClassTag[OUT]): Network[IN, OUT] = {
    val train = trainset.mapValues(expectConverter).setName(name + " (TRAIN)").persist(param.storageLevel)
    val test = testset.mapValues(expectConverter).setName(name + " (TRAIN)").persist(param.storageLevel)
    new StaticTrainer[IN, OUT](network, objective, train, test, name, param).getTrainedNetwork
  }

  def train[IN](network: Network[IN, DataVec],
                trainset: RDD[(IN, DataVec)],
                testset: RDD[(IN, DataVec)],
                objective: Objective,
                name: String = "temp")
               (implicit evidence$1: ClassTag[IN]): Network[IN, DataVec] = {
    val train = trainset.setName(name + " (TRAIN)").persist(param.storageLevel)
    val test = testset.setName(name + " (TRAIN)").persist(param.storageLevel)
    new StaticTrainer[IN, DataVec](network, objective, train, test, name, param).getTrainedNetwork
  }

  def trainFlexible[IN, OUT](network: Network[IN, OUT],
                             trainset: RDD[(IN, OUT)],
                             testset: RDD[(IN, OUT)],
                             objective: Objective,
                             expectConverter: OUT ⇒ DataVec,
                             name: String = "temp")
                            (implicit evidence$1: ClassTag[IN], evidence$2: ClassTag[OUT]): Network[IN, OUT] = {
    val train = trainset.setName(name + " (TRAIN)").persist(param.storageLevel)
    val test = testset.setName(name + " (TRAIN)").persist(param.storageLevel)
    new FlexibleTrainer[IN, OUT](network, objective, expectConverter, train, test, name, param).getTrainedNetwork
  }
}

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

  /**
   * Tail Recursive : Train each batch
   *
   * @param epoch current iteration epoch. (1 iteration = 1 validation freq)
   * @param patience current patience, i.e. loop until at least this epoch.
   * @return (Evaluation, Weight, Total) Loss when train is finished
   */
  @tailrec
  override protected final def trainSmallBatch(epoch: Int, patience: Int): Unit = {
    val trainSample = batchSampler.next()

    val input = trainSample.map(_._1).par
    val output = network.forward(input).seq

    val error = trainSample.map(_._2).zip(output).map {
      case (exp, out) ⇒ objective.derivative(exp, out)
    }

    network.backward(error)
    network.update(param.miniBatch)

    val maxEpoch = getMaxEpoch(epoch, patience)
    if (maxEpoch > 0)
      trainSmallBatch(epoch + 1, maxEpoch)
  }

  override protected def getValidationErr: Double = {
    val loss = testSet.context.accumulator(0.0)
    val count = testSet.context.accumulator(0)
    val nanCount = testSet.context.accumulator(0)
    testSet.foreachPartition {
      it ⇒
        var sum = 0.0
        var cnt = 0

        it.toStream.foreach {
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

    if (nanCount.value > 0)
      logger warn s"There was ${nanCount.value} NaN/Infinity value during Evaluation Process!"

    if (count.value == 0) {
      logger warn "There's no regular instance inside evaluation set!"
      Double.PositiveInfinity
    } else {
      loss.value / count.value
    }
  }
}

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

  /**
   * Tail Recursive : Train each batch
   *
   * @param epoch current iteration epoch. (1 iteration = 1 validation freq)
   * @param patience current patience, i.e. loop until at least this epoch.
   * @return (Evaluation, Weight, Total) Loss when train is finished
   */
  @tailrec
  override protected final def trainSmallBatch(epoch: Int, patience: Int): Unit = {
    val trainSample = batchSampler.next()

    val input = trainSample.map(_._1).par
    val output = network.forward(input).seq

    val error = trainSample.map(_._2).zip(output).map {
      case (exp, out) ⇒ objective.derivative(expectConverter(exp), out)
    }

    network.backward(error)
    network.update(param.miniBatch)

    val maxEpoch = getMaxEpoch(epoch, patience)
    if (maxEpoch > 0)
      trainSmallBatch(epoch + 1, maxEpoch)
  }

  override protected def getValidationErr: Double = {
    val loss = testSet.context.accumulator(0.0)
    val count = testSet.context.accumulator(0)
    val nanCount = testSet.context.accumulator(0)
    testSet.foreachPartition {
      it ⇒
        var sum = 0.0
        var cnt = 0

        it.toStream.foreach {
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

    if (nanCount.value > 0)
      logger warn s"There was ${nanCount.value} NaN/Infinity value during Evaluation Process!"

    if (count.value == 0) {
      logger warn "There's no regular instance inside evaluation set!"
      Double.PositiveInfinity
    } else {
      loss.value / count.value
    }
  }
}
