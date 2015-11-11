package com.github.nearbydelta.deepspark.train

import org.apache.spark.storage.StorageLevel

/**
 * __Criteria__: When to stop training
 *
 * This case class defines when to stop training. Training stops if one of the following condition is satisfied.
 *
- #Iteration ≥ maxIter
   - #Iteration ≥ current patience value, which is calculated by `max(patience, bestIteration * patienceStep)`
   - Amount of loss < lossThreshold
 *
 * Validation is done for each `validationFreq` iterations,
 * and whenever current/best loss ratio below improveThreshold,
 * that iteration is marked as best iteration.
 *
 * @param maxIter __maximum__ iteration count `(default 100,000)`
 * @param waitAfterUpdate __multiplier__ for calculating patience `(default 1 := Wait lastupdate# * 1 after update)`
 * @param improveThreshold __threshold__ that iteration is marked as "improved" `(default 99.5% = 0.995)`
 * @param lossThreshold __maximum-tolerant__ loss value. `(default 0.0001)`
 * @param validationFreq __multiplier__ used for count for validation. `(default 1.0)`
 *                       Validation checked whenever (validationFreq) * (#epoch for 1 training batch).
 *                       where #epoch for 1 iteration = round(1 / miniBatchFraction).
 * @param miniBatch __size__ of mini-batch. `(default 10)`
 * @param reuseSaveData True if this trainer reuse previous temp data in disk. `(default false)`
 * @param dataOnLocal True if want to collect all the data into the driver. `(default false)`
 * @param storageLevel Persist level `(default MEMORY_ONLY)`
 */
case class TrainingParam(maxIter: Int = 100,
                         waitAfterUpdate: Int = 1,
                         improveThreshold: Double = 0.999999,
                         lossThreshold: Double = 0.0001,
                         validationFreq: Double = 1.0,
                         miniBatch: Int = 10,
                         reuseSaveData: Boolean = false,
                         dataOnLocal: Boolean = false,
                         storageLevel: StorageLevel = StorageLevel.MEMORY_ONLY)
  extends Serializable {
  require(miniBatch < 1000, "Your mini-batch setup does not seems to be 'mini.'")
  require(maxIter > 10, "At least 10 iteration required.")
  require(validationFreq > 0.001, "We recommend validation frequency at least 0.1%.")
  require(waitAfterUpdate > 0, "patience(waitAfterUpdate) value must be set above zero.")
  require(improveThreshold < 1.0, "Improve threshold above 1.0 cause up-hill movement.")
  require(improveThreshold > 0.5, "We recommend improve threshold above 0.5")
}