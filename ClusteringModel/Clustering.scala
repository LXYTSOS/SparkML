package com.lxy.ml

import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.linalg.Vectors

/**
 * @author sl169
 */
object Clustering {
  def main(args: Array[String]){
    val sc = new SparkContext("local", "Clustering")
    
    //获取电影数据
    val movies = sc.textFile("F:\\ScalaWorkSpace\\data\\ml-100k\\u.item")
    println(movies.first())
    //1|Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0
    
    //获取电影的类型
    val genres = sc.textFile("F:\\ScalaWorkSpace\\data\\ml-100k\\u.genre")
    genres.take(5).foreach(println)
//    unknown|0
//    Action|1
//    Adventure|2
//    Animation|3
//    Children's|4
    
    //映射电影类型，如0 -> unknown, 1 -> Action
    val genreMap = genres.filter(!_.isEmpty()).map(line => line.
        split("\\|")).map(array => (array(1), array(0))).collectAsMap
    println(genreMap)
    //Map(2 -> Adventure, 5 -> Comedy, 12 -> Musical, 15 -> Sci-Fi, 8 -> Drama, 18 -> Western, 7 -> Documentary, 17 -> War, 1 -> Action, 4 -> Children's, 11 -> Horror, 14 -> Romance, 6 -> Crime, 0 -> unknown, 9 -> Fantasy, 16 -> Thriller, 3 -> Animation, 10 -> Film-Noir, 13 -> Mystery)
    
    //将电影名称与它对应的类型对应起来
    val titlesAndGenres = movies.map(_.split("\\|")).map{ array => 
      val genres = array.toSeq.slice(5, array.size)
      val genresAssigned = genres.zipWithIndex.filter{ case (g, idx)
        =>
          g == "1"
      }.map{ case (g, idx) => 
        genreMap(idx.toString())
      }
      (array(0).toInt, (array(1), genresAssigned))
    }
    println(titlesAndGenres.first())
    //(1,(Toy Story (1995),ArrayBuffer(Animation, Children's, Comedy)))
    
    val rawData = sc.textFile("F:\\ScalaWorkSpace\\data\\ml-100k\\u.data")
    val rawRatings = rawData.map(_.split("\t").take(3))
    val ratings = rawRatings.map { case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble) }
    ratings.cache
    val alsModel = ALS.train(ratings, 50, 10, 0.1)
    
    val movieFactors = alsModel.productFeatures.map{ case (id, factor) => (id, Vectors.dense(factor))}
    val movieVectors = movieFactors.map(_._2)
    val userFactors = alsModel.userFeatures.map{ case (id, factor) => (id, Vectors.dense(factor))}
    val userVectors = userFactors.map(_._2)
    
  }
}