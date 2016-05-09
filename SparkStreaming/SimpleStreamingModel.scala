package com.lxy.sparkstreaming

import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.Seconds
import breeze.linalg.DenseVector
import org.apache.spark.mllib.regression.StreamingLinearRegressionWithSGD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * @author sl169
 * 一个简单的线性回归计算出每个批次的预测值
 */
object SimpleStreamingModel {
  def main(args: Array[String]){
    val ssc = new StreamingContext("local[*]","First Streaming App", Seconds(10))
    val stream = ssc.socketTextStream("localhost", 9999)
    
    val NumFeatures = 100
    val zeroVector = DenseVector.zeros[Double](NumFeatures)
    val model = new StreamingLinearRegressionWithSGD()
      .setInitialWeights(Vectors.dense(zeroVector.data))
      .setNumIterations(1)
      .setStepSize(0.01)
    
    //创建一个标签点的流
    val labeledStream = stream.map{ event => 
      val split = event.split("\t")
      val y = split(0).toDouble
      val features = split(1).split(",").map(_.toDouble)
      LabeledPoint(label = y, features = Vectors.dense(features))
    }
    
    //在流上训练测试模型，并打印预测结果
    model.trainOn(labeledStream)
    model.predictOn(labeledStream.map(ls=>ls.features)).print()
    
    ssc.start()
    ssc.awaitTermination()
  }
}