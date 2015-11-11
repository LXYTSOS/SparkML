package com.lxy.ml

import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.recommendation.ALS
import org.jblas.DoubleMatrix

//This is the second commit test
object ExtractFeatures {

  def cosineSimilarity(vec1: DoubleMatrix, vec2: DoubleMatrix): Double = {
    vec1.dot(vec2) / (vec1.norm2() * vec2.norm2())
  }

  def main(args: Array[String]) {
    val sc = new SparkContext("local", "ExtractFeatures")
    //    System.setProperty("spark.executor.memory", "1G")
    val rawData = sc.textFile("F:\\ScalaWorkSpace\\data\\ml-100k\\u.data")
//        println(rawData.first())

    val rawRatings = rawData.map(_.split("\t").take(3))
//        rawRatings.first().foreach(println)

    val ratings = rawRatings.map {
      case Array(user, movie, rating) =>
        Rating(user.toInt, movie.toInt, rating.toDouble)
    }
//        println(ratings.first())

    val model = ALS.train(ratings, 50, 8, 0.01)
    //迭代9次开始出现栈内存溢出，所以把迭代参数改成8次
//    println(model.userFeatures.count)
//        println(model.productFeatures.count)

    val predictedRating = model.predict(789, 123) //user789,movie123
//        println(predictedRating)

    val userID = 789
    val K = 10
    val topKRecs = model.recommendProducts(userID, K);
//    println(topKRecs.mkString("\n"))

    //取到电影名字
    val movies = sc.textFile("F:\\ScalaWorkSpace\\data\\ml-100k\\u.item")
    val titles = movies.map(line => line.split("\\|").take(2)).map(array => (array(0).toInt, array(1))).collectAsMap()
//        println(titles(123))

    val moviesForUser = ratings.keyBy(_.user).lookup(789)
//    println(moviesForUser.size)

    //取到此用户前10个电影名字和分数
//    moviesForUser.sortBy(-_.rating).take(10).map(rating => (titles(rating.product), rating.rating)).foreach(println)

    //取得为此用户推荐的前10个电影的名字和分数
//    topKRecs.map(rating => (titles(rating.product), rating.rating)).foreach(println)
    
    val itemId = 567
    val itemFactor = model.productFeatures.lookup(itemId).head
    val itemVector = new DoubleMatrix(itemFactor)
//    println(cosineSimilarity(itemVector, itemVector))
    
    val sims = model.productFeatures.map{ case (id, factor) => 
      val factorVector = new DoubleMatrix(factor)
      val sim = cosineSimilarity(factorVector, itemVector)
      (id,sim)
    }
    
    val sortedSims = sims.top(K)(Ordering.by[(Int, Double), Double]{
      case(id, similarity) => similarity
    })
    
//    println(sortedSims.take(10).mkString("\n"))
    
    val sortedSims2 = sims.top(K+1)(Ordering.by[(Int, Double), Double]{
      case(id, similarity) => similarity
    })
    
//    println(sortedSims2.slice(1, 11).map{case (id, sim) => (titles(id), sim)}.mkString("\n"))
    
//    val actualRating = moviesForUser.take(1)(0)
//    println(actualRating)
//    val predictedRating1 = model.predict(789, actualRating.product)
//    println(predictedRating1)
//    val squaredError = math.pow(predictedRating1 - actualRating.rating, 2.0)
//    println(squaredError)
    
    //下面部分代码会导致堆栈溢出，原因电脑内存不够，-Xmx4096m-Xms4000m-XX:MaxPermSize=4024m
//    val usersProducts = ratings.map{case Rating(user, product, rating) => (user, product)}
//    val predictions = model.predict(usersProducts).map{case Rating(user, product, rating) => ((user, product), rating)}
//    
//    val ratingsAndPredictions = ratings.map{case Rating(user, product, rating) => ((user, product), rating)}.join(predictions)
//    
//    val MSE = ratingsAndPredictions.map{case ((user,product), (actual, predicted)) => math.pow((actual - predicted), 2.0)
//      }.reduce(_ + _) / ratingsAndPredictions.count
//    println("Mean Squared Error = " + MSE)
//    val RMSE = math.sqrt(MSE)
//    println("Root Mean Squared Error = " + RMSE)
  }
}