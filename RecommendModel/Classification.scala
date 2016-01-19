package com.lxy.ml

import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.optimization.Updater
import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.optimization.SimpleUpdater

/**
 * @author sl169
 */
object Classification {
  def main(args: Array[String]) {
    val sc = new SparkContext("local", "Classification")

    val rawData = sc.textFile("F:\\ScalaWorkSpace\\data\\train_noheader.tsv")
    val records = rawData.map(line => line.split("\t"))
    //    records.first().foreach(println)

    val data = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(4, r.size - 1).map(d => if (d ==
        "?") 0.0 else d.toDouble)
      LabeledPoint(label, Vectors.dense(features))
    }
    data.cache
    //    val numData = data.count
    //    println(numData)

    val nbData = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(4, r.size - 1).map(d => if (d ==
        "?") 0.0 else d.toDouble).map(d => if (d < 0) 0.0 else d)
      LabeledPoint(label, Vectors.dense(features))
    }

    val numIterations = 10
    val maxTreeDepth = 5

    //训练逻辑回归模型
    val lrModel = LogisticRegressionWithSGD.train(data, numIterations)

    //训练SVM模型
    val svmModel = SVMWithSGD.train(data, numIterations)

    //训练朴素贝叶斯模型
    val nbModel = NaiveBayes.train(nbData)

    //训练决策树模型
    val dtModel = DecisionTree.train(data, Algo.Classification, Entropy, maxTreeDepth)

    val dataPoint = data.first
    val prediction = lrModel.predict(dataPoint.features)
    //    println(prediction)  1.0
    val trueLabel = dataPoint.label
    //    println(trueLabel)  0.0
    //So, in this case, our model got it wrong!

    val predictions = lrModel.predict(data.map(lp => lp.features))
    //    predictions.take(5).foreach(println)

    val numData = data.count
    /*val lrTotalCorrect = data.map { point =>
      if (lrModel.predict(point.features) == point.label) 1 else 0
      }.sum

    val svmTotalCorrect = data.map { point =>
      if (svmModel.predict(point.features) == point.label) 1 else 0
    }.sum

    val nbTotalCorrect = nbData.map { point =>
      if (nbModel.predict(point.features) == point.label) 1 else 0
    }.sum

    val dtTotalCorrect = data.map { point =>
      val score = dtModel.predict(point.features)
      val predicted = if (score > 0.5) 1 else 0
      if (predicted == point.label) 1 else 0
    }.sum

    
    val lrAccuracy = lrTotalCorrect / numData
    val svmAccuracy = svmTotalCorrect / numData
    val nbAccuracy = nbTotalCorrect / numData
    val dtAccuracy = dtTotalCorrect / numData

    println("逻辑回归准确率：" + lrAccuracy)
    println("SVM准确率：" + svmAccuracy)
    println("朴素贝叶斯准确率：" + nbAccuracy)
    println("决策树准确率：" + dtAccuracy)*/

    //PR和ROC曲线下面积
    /*val metrics = Seq(lrModel, svmModel).map{model =>
      val scoreAndLabels = data.map{point =>
        (model.predict(point.features), point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.
    areaUnderROC)
    }
    
    val nbMetrics = Seq(nbModel).map{ model =>
      val scoreAndLabels = nbData.map { point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR,
      metrics.areaUnderROC)
    }
    
    val dtMetrics = Seq(dtModel).map{ model =>
      val scoreAndLabels = data.map { point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR,
      metrics.areaUnderROC)
    }
    val allMetrics = metrics ++ nbMetrics ++ dtMetrics
    allMetrics.foreach{ case (m, pr, roc) =>
      println(f"$m, Area under PR: ${pr * 100.0}%2.4f%%, Area under ROC: ${roc * 100.0}%2.4f%%")
    }*/

    val vectors = data.map(lp => lp.features)
    //    val matrix = new RowMatrix(vectors)
    //    val matrixSummary = matrix.computeColumnSummaryStatistics()
    //    println(matrixSummary.mean)
    //    println(matrixSummary.min)
    //    println(matrixSummary.max)
    //    println(matrixSummary.variance)
    //    println(matrixSummary.numNonzeros)

    val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
    val scaledData = data.map(lp => LabeledPoint(lp.label,
      scaler.transform(lp.features)))

    //    println(data.first.features)
    //    println(scaledData.first.features)

    //    val lrModelScaled = LogisticRegressionWithSGD.train(scaledData, numIterations)
    //    
    //    val lrTotalCorrectScaled = scaledData.map { point =>
    //      if (lrModelScaled.predict(point.features) == point.label) 1 else
    //      0
    //    }.sum
    //    val lrAccuracyScaled = lrTotalCorrectScaled / numData
    //    val lrPredictionsVsTrue = scaledData.map { point =>
    //      (lrModelScaled.predict(point.features), point.label)
    //    }
    //    val lrMetricsScaled = new BinaryClassificationMetrics(lrPredictionsVsTrue)
    //    val lrPr = lrMetricsScaled.areaUnderPR
    //    val lrRoc = lrMetricsScaled.areaUnderROC
    //    println(f"${lrModelScaled.getClass.getSimpleName}\nAccuracy:${lrAccuracyScaled * 100}%2.4f%%\nArea under PR: ${lrPr *100.0}%2.4f%%\nArea under ROC: ${lrRoc * 100.0}%2.4f%%")

    //加入新特征，内容属于什么方面    
    val categories = records.map(r => r(3)).distinct.collect.zipWithIndex.toMap
    val numCategories = categories.size
    //    println(categories)
    //    println(numCategories)

    val dataCategories = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val categoryIdx = categories(r(3))
      val categoryFeatures = Array.ofDim[Double](numCategories)
      categoryFeatures(categoryIdx) = 1.0
      val otherFeatures = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
      val features = categoryFeatures ++ otherFeatures
      LabeledPoint(label, Vectors.dense(features))
    }
    //    println(dataCategories.first)

    val scalerCats = new StandardScaler(withMean = true, withStd = true).
      fit(dataCategories.map(lp => lp.features))
    val scaledDataCats = dataCategories.map(lp =>
      LabeledPoint(lp.label, scalerCats.transform(lp.features)))
    scaledDataCats.cache
    //        println(dataCategories.first.features)
    //        println(scaledDataCats.first.features)

    //使用正确的数据格式提高模型准确率
    //    val dataNB = records.map { r =>
    //      val trimmed = r.map(_.replaceAll("\"", ""))
    //      val label = trimmed(r.size - 1).toInt
    //      val categoryIdx = categories(r(3))
    //      val categoryFeatures = Array.ofDim[Double](numCategories)
    //      categoryFeatures(categoryIdx) = 1.0
    //      LabeledPoint(label, Vectors.dense(categoryFeatures))
    //    }
    //
    //    val nbModelCats = NaiveBayes.train(dataNB)
    //    val nbTotalCorrectCats = dataNB.map { point =>
    //      if (nbModelCats.predict(point.features) == point.label) 1 else 0
    //    }.sum
    //    val nbAccuracyCats = nbTotalCorrectCats / numData
    //    val nbPredictionsVsTrueCats = dataNB.map { point =>
    //      (nbModelCats.predict(point.features), point.label)
    //    }
    //    val nbMetricsCats = new BinaryClassificationMetrics(nbPredictionsVsTrueCats)
    //    val nbPrCats = nbMetricsCats.areaUnderPR
    //    val nbRocCats = nbMetricsCats.areaUnderROC
    //    println(f"${nbModelCats.getClass.getSimpleName}\nAccuracy:${nbAccuracyCats * 100}%2.4f%%\nArea under PR: ${nbPrCats * 100.0}%2.4f%%\nArea under ROC: ${nbRocCats * 100.0}%2.4f%%")
    
    //迭代次数
//    val iterResults = Seq(1, 5, 10, 50).map { param =>
//      val model = trainWithParams(scaledDataCats, 0.0, param, new SimpleUpdater, 1.0)
//      createMetrics(s"$param iterations", scaledDataCats, model)
//    }
//    iterResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
    
    //步长
    val stepResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = trainWithParams(scaledDataCats, 0.0, numIterations, new
    SimpleUpdater, param)
      createMetrics(s"$param step size", scaledDataCats, model)
    }
    stepResults.foreach { case (param, auc) => println(f"$param, AUC =${auc * 100}%2.2f%%") }

  }

  def trainWithParams(input: RDD[LabeledPoint], regParam: Double,
                      numIterations: Int, updater: Updater, stepSize: Double) = {
    val lr = new LogisticRegressionWithSGD
    lr.optimizer.setNumIterations(numIterations).
      setUpdater(updater).setRegParam(regParam).setStepSize(stepSize)
    lr.run(input)
  }

  def createMetrics(label: String, data: RDD[LabeledPoint], model: ClassificationModel) = {
    val scoreAndLabels = data.map { point =>
      (model.predict(point.features), point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (label, metrics.areaUnderROC)
  }
}