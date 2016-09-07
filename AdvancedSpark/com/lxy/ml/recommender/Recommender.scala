package com.lxy.ml.recommender

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.recommendation.ALS

object Recommender {
  def main(args: Array[String]){
    val conf = new SparkConf()
      .setAppName("Recommender")
      .setMaster("local[*]")
      .set("spark,driver.memory", "2G")
     
    val sc = new SparkContext(conf)
    
    //用户ID－歌手ID
    val rawUserArtistData = sc.textFile("src/data/recommender/user_artist_data.txt")
    
//    val user = rawUserArtistData.map(_.split(" ")(0).toDouble).stats()
//    val artist = rawUserArtistData.map(_.split(" ")(1).toDouble).stats()
//    
//    println(user.max)
//    println(artist.max)
    
    //歌手ID－歌手名
    val rawArtistData = sc.textFile("src/data/recommender/artist_data.txt")
//    val artistByID = rawArtistData.map{ line =>
//      val id = line.split("\t")(0).toInt
//      val name = line.split("\t")(1).trim()
//      (id,name)
//    }
    
//    val artistByID = rawArtistData.map{ line =>
//      val (id, name) = line.span(_ != '\t')
//      (id.toInt, name.trim())
//    }
    
    val artistByID = rawArtistData.flatMap { line =>  
      val (id, name) = line.span(_ != '\t')
      if(name.isEmpty()){
        None
      }else{
        try{
          Some((id.toInt, name.trim()))
        }catch{
          case e: NumberFormatException => None
        }
      }
    }
    
    //歌手别名
    val rawArtistAlias = sc.textFile("src/data/recommender/artist_alias.txt")
    val artistAlias = rawArtistAlias.flatMap { line => 
      val tokens = line.split('\t')
      if(tokens(0).isEmpty()){
        None
      }else{
        Some((tokens(0).toInt, tokens(1).toInt))
      }
    }.collectAsMap()
    
    val bArtistAlias = sc.broadcast(artistAlias)
    
    val trainData = rawUserArtistData.map{ line =>
      val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
      val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
      Rating(userID, finalArtistID, count)
    }.cache()
    
    val model = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)
    
    //找到一个用户听过的所有歌手ID
    val rawArtistForUser = rawUserArtistData.map(_.split(" ")).filter{ case Array(user,_,_) => user.toInt == 2093760}
    
    //将这个用户听过的歌手ID转换成不重复的Set
    val existingProducts = rawArtistForUser.map{ case Array(_, artist, _) => artist.toInt}.collect().toSet
    
    artistByID.filter{ case (id, name) => existingProducts.contains(id)}.values.collect().foreach(println)
    
    val recommendations = model.recommendProducts(2093760, 5)
    recommendations.foreach(println)
    
    val recommendedProductIDs = recommendations.map(_.product).toSet
    
    artistByID.filter{ case (id, name) => recommendedProductIDs.contains(id)}.values.collect().foreach(println)
    
  }
}