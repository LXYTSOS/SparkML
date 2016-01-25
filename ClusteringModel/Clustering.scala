package com.lxy.ml

import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.clustering.KMeans
import breeze.linalg.DenseVector
import breeze.numerics.pow

/**
 * @author sl169
 */
object Clustering {
  //Euclidean distances
  def computeDistance(v1: DenseVector[Double], v2: DenseVector[Double]) = pow(v1 - v2, 2).sum
  
  def main(args: Array[String]){
    val sc = new SparkContext("local", "Clustering")
    
    //获取电影数据
    val movies = sc.textFile("F:\\ScalaWorkSpace\\data\\ml-100k\\u.item")
//    println(movies.first())
    //1|Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0
    
    //获取电影的类型
    val genres = sc.textFile("F:\\ScalaWorkSpace\\data\\ml-100k\\u.genre")
//    genres.take(5).foreach(println)
//    unknown|0
//    Action|1
//    Adventure|2
//    Animation|3
//    Children's|4
    
    //映射电影类型，如0 -> unknown, 1 -> Action
    val genreMap = genres.filter(!_.isEmpty()).map(line => line.
        split("\\|")).map(array => (array(1), array(0))).collectAsMap
//    println(genreMap)
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
//    println(titlesAndGenres.first())
    //(1,(Toy Story (1995),ArrayBuffer(Animation, Children's, Comedy)))
    
    val rawData = sc.textFile("F:\\ScalaWorkSpace\\data\\ml-100k\\u.data")
    val rawRatings = rawData.map(_.split("\t").take(3))
    val ratings = rawRatings.map { case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble) }
    ratings.cache
    val alsModel = ALS.train(ratings, 50, 7, 0.1)
    
    val movieFactors = alsModel.productFeatures.map{ case (id, factor) => (id, Vectors.dense(factor))}
    val movieVectors = movieFactors.map(_._2)
    val userFactors = alsModel.userFeatures.map{ case (id, factor) => (id, Vectors.dense(factor))}
    val userVectors = userFactors.map(_._2)
    
    //正则化
//    val movieMatrix = new RowMatrix(movieVectors)
//    val movieMatrixSummary = movieMatrix.computeColumnSummaryStatistics()
//    val userMatrix = new RowMatrix(userVectors)
//    val userMatrixSummary = userMatrix.computeColumnSummaryStatistics()
//    println("Movie factors mean: " + movieMatrixSummary.mean)
//    println("Movie factors variance: " + movieMatrixSummary.variance)
//    println("User factors mean: " + userMatrixSummary.mean)
//    println("User factors variance: " + userMatrixSummary.variance)
    
    //训练聚类模型
    movieVectors.cache()
    userVectors.cache()
    //The input data was not directly cached, which may hurt performance if its parent RDDs are also uncached.
    val numClusters = 5
    val numIterations = 10
    val numRuns = 3
    
    val movieClusterModel = KMeans.train(movieVectors, numClusters, numIterations, numRuns)
//    val movieClusterModelConverged = KMeans.train(movieVectors, numClusters, 100)
    val userClusterModel = KMeans.train(userVectors, numClusters, numIterations, numRuns)
    
    //传入单个值进行预测
    val movie1 = movieVectors.first()
    val movieCluster = movieClusterModel.predict(movie1)
//    println(movieCluster)
    // 1, Different runs map have different results
    
    //传入RDD进行预测
    val predictions = movieClusterModel.predict(movieVectors)
//    println(predictions.take(10).mkString(","))
    // 1,2,4,1,2,0,0,1,0,0, Different runs map have different results
    
    //Interpreting the movie clusters
//    val titlesWithFactors = titlesAndGenres.join(movieFactors)
//    val moviesAssigned = titlesWithFactors.map{ 
//      case (id, ((title, genres), vector)) => 
//        val pred = movieClusterModel.predict(vector)
//        val clusterCenter = movieClusterModel.clusterCenters(pred)
//        val dist = computeDistance(DenseVector(clusterCenter.toArray), DenseVector(vector.toArray))
//        (id, title, genres.mkString(" "), pred, dist)
//    }
//    val clusterAssignments = moviesAssigned.groupBy{ case (id, title, genres, cluster, dist) => 
//      cluster 
//    }.collectAsMap
    
//    for( (k, v) <-clusterAssignments.toSeq.sortBy(_._1)){
//      println(s"Cluster $k:")
//      val m = v.toSeq.sortBy(_._5)
//      println(m.take(20).map{ case (_, title, genre, _, d) => 
//        (title, genre, d)}.mkString("\n"))
//      println("============\n")
//    }
    
    //Compute performance metrics
//    val movieCost = movieClusterModel.computeCost(movieVectors)
//    val userCost = userClusterModel.computeCost(userVectors)
//    println("WCSS for movies: " + movieCost)
//    println("WCSS for users: " + userCost)
    
    //Compute cross-validation metrics for movies
//    val trainTestSplitMovies = movieVectors.randomSplit(Array(0.6,0.4), 123)
//    val trainMovies = trainTestSplitMovies(0)
//    val testMovies = trainTestSplitMovies(1)
//    val costsMovies = Seq(2,3,4,5,10,20).map{ k => 
//      (k, KMeans.train(trainMovies, numIterations, k, numRuns).computeCost(testMovies))
//    }
//    println("Movie clustering cross-validation:")
//    costsMovies.foreach{ case (k, cost) => 
//      println(f"WCSS for K=$k id $cost%2.2f")
//    }
    
    //Compute cross-validation metrics for users
    val trainTestSplitUsers = userVectors.randomSplit(Array(0.6,0.4), 123)
    val trainUsers = trainTestSplitUsers(0)
    val testUsers = trainTestSplitUsers(1)
    val costsUsers = Seq(2,3,4,5,10,20).map { k => 
      (k, KMeans.train(trainUsers, numIterations, k, numRuns).computeCost(testUsers)) 
    }
    println("User clustering cross-validation:")
    costsUsers.foreach{ case (k, cost) => 
      println(f"WCSS for K=$k id $cost%2.2f")
    }
  }
}