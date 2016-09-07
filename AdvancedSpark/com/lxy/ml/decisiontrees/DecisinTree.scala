package com.lxy.ml.decisiontrees

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.tree.DecisionTree

object DecisinTree {
  def main(args: Array[String]){
    val conf = new SparkConf()
      .setAppName("DecisionTree")
      .setMaster("local[*]")
      .set("spark.driver.memory", "2G")
     
    val sc = new SparkContext(conf)
    
    val rawData = sc.textFile("src/data/decisointress/covtype.data")
//    val data = rawData.map { line => 
//      val values = line.split(',').map(_.toDouble)
//      //init returns all but last value; target is last column
//      val featureVector = Vectors.dense(values.init)
//      //DecisionTree needs labels starting at 0; subtract 1
//      val label = values.last - 1
//      LabeledPoint(label, featureVector)
//    }
    
    val data = rawData.map { line =>
      val values = line.split(',').map(_.toDouble)
      val wilderness = values.slice(10, 14).indexOf(1.0).toDouble
      val soil = values.slice(14, 54).indexOf(1.0).toDouble
      val featureVector = Vectors.dense(values.slice(0,10) :+ wilderness :+ soil)
      val label = values.last - 1
      LabeledPoint(label, featureVector)
    }
    
    //下面开始分割数据集，80%用来训练，10%用来交叉验证（选择最佳超参数），10%用来测试（对使用最佳超参数训练出的模型进行无偏估计）
    val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))
    
    trainData.cache()
    cvData.cache()
    testData.cache()
    
//    val model = DecisionTree.trainClassifier(trainData, 7, Map[Int, Int](), "gini", 4, 100)
    
//    val metrics = getMetrics(model, cvData)
//    println(metrics.confusionMatrix)
//    println(metrics.precision)
//    (0 until 7).map(
//        cat => (metrics.precision(cat), metrics.recall(cat))
//    ).foreach(println)
    
//    val trainPriorProbabilities = classProbabilities(trainData)
//    
//    val cvPriorProbabilities = classProbabilities(cvData)
//    
//    val probability = trainPriorProbabilities.zip(cvPriorProbabilities).map{
//      case(trainProb, cvProb) => trainProb * cvProb
//    }.sum
//    println(probability)
    
//    val evaluations = 
//      for (impurity <- Array("gini", "entropy");
//          depth     <- Array(1, 20);
//          bins      <- Array(10, 300))
//        yield {
//        val model = DecisionTree.trainClassifier(trainData, 7, Map[Int, Int](), impurity, depth, bins)
//        val predictionsAndLabels = cvData.map(example =>
//          (model.predict(example.features), example.label)
//        )
//        val accuracy = new MulticlassMetrics(predictionsAndLabels).precision
//        ((impurity, depth, bins), accuracy)
//      }
//    
//    evaluations.sortBy(-_._2).foreach(println)
    
    val evaluations = 
      for (impurity <- Array("gini", "entropy");
          depth     <- Array(10, 20, 30);
          bins      <- Array(40, 300))
        yield {
        val model = DecisionTree.trainClassifier(trainData, 7, Map(10 -> 4, 11 -> 40), impurity, depth, bins)
        val trainAccuracy = getMetrics(model, trainData).precision
        val cvAccuracy = getMetrics(model, cvData).precision
        ((impurity, depth, bins), (trainAccuracy, cvAccuracy))
      }
    
    evaluations.sortBy(-_._2._1).foreach(println)
  }
  
  def classProbabilities(data: RDD[LabeledPoint]): Array[Double] = {
    val countsByCategory = data.map(_.label).countByValue()
    val counts = countsByCategory.toArray.sortBy(_._1).map(_._2)
    counts.map(_.toDouble / counts.sum)
  }
  
  def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
    val predictionsAndLabels = data.map(example => 
      (model.predict(example.features), example.label)
    )
    new MulticlassMetrics(predictionsAndLabels)
  }
}