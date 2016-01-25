package com.lxy.ml

import org.apache.spark.SparkContext

/**
 * @author sl169
 */
object DimensionalityReduction {
  def main(args: Array[String]){
    val sc = new SparkContext("local", "DimensionalityReduction")
//    System.setProperty("spark.executor.memory", "2G")
    
//    val path = "facialData/lfw/*"
    
    //在Windows下路径需要这么写：末尾通配符为*\*，lfw目录下所有文件夹下所有文件，而在Linux上则不需要这么写，直接写成lfw/*就行，如上
    val path = "F:\\ScalaWorkSpace\\data\\lfw\\*\\*"
    val rdd =  sc.wholeTextFiles(path)
    val first = rdd.first
    println(first)
  }
}