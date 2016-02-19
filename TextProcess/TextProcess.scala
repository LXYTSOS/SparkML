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
    
    val whiteSpaceSplit = text.flatMap(t => t.split(" ").map(_.toLowerCase))
//    println(whiteSpaceSplit.distinct().count())
    //402978,花了20分钟得出结果......
    
//    println(whiteSpaceSplit.sample(true, 0.3, 42).take(100).mkString(","))
    
    val nonWordSplit = text.flatMap(t => t.split("""\W+""").map(_.toLowerCase))
    //截取出字母、数字、下划线
//    println(nonWordSplit.distinct.count)
    //130126
//    println(nonWordSplit.sample(true, 0.3, 42).take(100).mkString(","))
    
    val regex = """[^0-9]*""".r
    val filterNumbers = nonWordSplit.filter( token => regex.pattern.matcher(token).matches())
//    println(filterNumbers.distinct().count())
    //84912
//    println(filterNumbers.sample(true, 0.3, 42).take(100).mkString(","))
    
    val tokenCounts = filterNumbers.map(t => (t,1)).reduceByKey(_ + _)
    val oreringDesc = Ordering.by[(String, Int), Int](_._2)
//    println(tokenCounts.top(20)(oreringDesc).mkString("\n"))
    
    //创建常用词集
    val stopwords = Set(
      "the","a","an","of","or","in","for","by","on","but", "is", "not",
      "with", "as", "was", "if",
      "they", "are", "this", "and", "it", "have", "from", "at", "my",
      "be", "that", "to"
    )
    
    //移除常用词
    val tokenCountsFilteredStopwords = tokenCounts.filter{ case (k, v) => !stopwords.contains(k)}
//    println(tokenCountsFilteredStopwords.top(20)(oreringDesc).mkString("\n"))
    
    //移除长度为一的
    val tokenCountsFilteredSize = tokenCountsFilteredStopwords.filter{ case (k, v) => k.size >= 2 }
//    println(tokenCountsFilteredSize.top(20)(oreringDesc).mkString("\n"))
    
    val oreringAsc = Ordering.by[(String, Int), Int](-_._2)
//    println(tokenCountsFilteredSize.top(20)(oreringAsc).mkString("\n"))
    
    val rareTokens = tokenCounts.filter{ case (k, v) => v < 2}.map{ case (k, v) => k}.collect.toSet
    val tokenCountsFilteredAll = tokenCountsFilteredSize.filter{case (k,v) => !rareTokens.contains(k)}
//    println(tokenCountsFilteredAll.top(20)(oreringAsc).mkString("\n"))
//    println(tokenCountsFilteredAll.count)
    //51801
  }
}