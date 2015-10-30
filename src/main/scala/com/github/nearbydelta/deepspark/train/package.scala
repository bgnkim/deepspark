package com.github.nearbydelta.deepspark

import org.apache.spark.storage.StorageLevel

/**
 * Created by bydelta on 15. 10. 16.
 */
package object train {

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
   * @param maxIter __maximum mini-batch__ iteration count `(default 100,000)`
   * @param waitAfterUpdate __multiplier__ for calculating patience `(default 1 := Wait lastupdate# * 1 after update)`
   * @param improveThreshold __threshold__ that iteration is marked as "improved" `(default 99.5% = 0.995)`
   * @param lossThreshold __maximum-tolerant__ loss value. `(default 0.0001)`
   * @param validationFreq __multiplier__ used for count for validation. `(default 1.0)`
   *                       Validation checked whenever (validationFreq) * (#epoch for 1 training batch).
   *                       where #epoch for 1 iteration = round(1 / miniBatchFraction).
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
    extends Serializable

}
