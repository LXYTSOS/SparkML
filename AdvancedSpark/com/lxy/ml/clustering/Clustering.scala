package com.lxy.ml.clustering

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.clustering.KMeans

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
    val clusterLabelCount = labelsAndData.map{ case (label,datum) =>
      val cluster = model.predict(datum)
      (cluster, label)
    }.countByValue
    
    clusterLabelCount.toSeq.sorted.foreach{
      case ((cluster, label), count) =>
        println(f"$cluster%1s$label%18s$count%8s")
    }
  }
}