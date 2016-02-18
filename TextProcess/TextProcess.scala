package com.lxy.ml

import org.apache.spark.SparkContext

/**
 * @author sl169
 */
object TextProcess {
  def main(args: Array[String]){
    val sc = new SparkContext("local", "TextProcess")
    
    val path = "F:/ScalaWorkSpace/data/TextProcess/20news-bydate-train/*/*"
    val rdd = sc.wholeTextFiles(path)
    val text = rdd.map{ case (file,text) => text}
//    println(text.count)
    
    val newsgroups = rdd.map{ case (file, text) => 
      file.split("/").takeRight(2).head}
    val countByGroup = newsgroups.map(n => (n,1)).reduceByKey(_ + _).collect.sortBy(-_._2).mkString("\n")
//    println(countByGroup)
//    (rec.sport.hockey,600)
//    (soc.religion.christian,599)
//    (rec.motorcycles,598)
//    (rec.sport.baseball,597)
//    (sci.crypt,595)
//    (sci.med,594)
//    (rec.autos,594)
//    (sci.space,593)
//    (comp.windows.x,593)
//    (sci.electronics,591)
//    (comp.os.ms-windows.misc,591)
//    (comp.sys.ibm.pc.hardware,590)
//    (misc.forsale,585)
//    (comp.graphics,584)
//    (comp.sys.mac.hardware,578)
//    (talk.politics.mideast,564)
//    (talk.politics.guns,546)
//    (alt.atheism,480)
//    (talk.politics.misc,465)
//    (talk.religion.misc,377)
  }
}