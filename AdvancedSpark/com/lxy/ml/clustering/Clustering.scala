package com.lxy.ml.clustering

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.rdd.RDD

object Clustering {
  def main(args: Array[String]){
    val conf = new SparkConf()
      .setAppName("Clustering")
      .setMaster("local[*]")
      .set("spark.driver.memory", "2G")
    
    val sc = new SparkContext(conf)
    
    val rawData = sc.textFile("src/data/clustering/kddcup.data.corrected")
    
    val labelsAndData = rawData.map{ line =>
      val buffer = line.split(',').toBuffer
      buffer.remove(1,3)
      val label = buffer.remove(buffer.length - 1)
      val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
      (label, vector)
    }
    
    val data = labelsAndData.values.cache()
    
    val kmeans = new KMeans()
    val model = kmeans.run(data)
    
//    model.clusterCenters.foreach(println)
    
    //下面看下每个簇都有哪些类
//    val clusterLabelCount = labelsAndData.map{ case (label,datum) =>
//      val cluster = model.predict(datum)
//      (cluster, label)
//    }.countByValue
//    
//    clusterLabelCount.toSeq.sorted.foreach{
//      case ((cluster, label), count) =>
//        println(f"$cluster%1s$label%18s$count%8s")
//    }
    
    (5 to 40 by 5).map(k => (k, clusteringScore(data, k))).foreach(println)
  }
  
  //计算两个向量之间的距离
  def distance(a: Vector, b: Vector) = {
    math.sqrt(a.toArray.zip(b.toArray).map(p => p._1 - p._2).map(d => d*d).sum)
  }
  
  //计算一个向量到所属簇心的距离
  def distToCentroid(datum: Vector, model: KMeansModel) = {
    val cluster = model.predict(datum)
    val centroid = model.clusterCenters(cluster)
    distance(centroid, datum)
  }
  
  //计算数据点到簇心的平均距离
  def clusteringScore(data: RDD[Vector], k: Int) = {
    val kmeans = new KMeans()
    kmeans.setK(k)
    val model = kmeans.run(data)
    data.map(datum => distToCentroid(datum, model)).mean()
  }
}