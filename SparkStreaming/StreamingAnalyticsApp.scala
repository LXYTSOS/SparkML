package com.lxy.sparkstreaming

import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.Seconds
import java.text.SimpleDateFormat
import java.util.Date
import org.apache.log4j.Logger

/**
 * @author sl169
 */
object StreamingAnalyticsApp {
  val logger = Logger.getLogger("Streaming")
  def main(args: Array[String]){
    val ssc = new StreamingContext("local[2]", "FirstStreamingApp", Seconds(10))
    val stream = ssc.socketTextStream("localhost", 9999)
    
    val events = stream.map{ record => 
      val event = record.split(",")
      (event(0), event(1), event(2))
    }
    
    events.foreachRDD{ (rdd, time) => 
      val numPurchase = rdd.count()
      val uniqueUsers = rdd.map{case (user, _, _) => user}.distinct().count()
      val totalRevenue = rdd.map{case (_, _, price) => price.toDouble}.sum()
      val productByPopularity = rdd.map{case(user,product,price) => (product, 1)}
        .reduceByKey(_+_)
        .collect()
        .sortBy(-_._2)
      val mostPopular = productByPopularity(0)
      
      val formatter = new SimpleDateFormat
      val dateStr = formatter.format(new Date((time.milliseconds)))
      logger.info(s"== Batch start time: $dateStr ==")
      logger.info("Total purchases: "+numPurchase)
      logger.info("Unique users: "+uniqueUsers)
      logger.info("Total revenue: "+totalRevenue)
      logger.info("Most popular product: %s with %d purchases".format(mostPopular._1, mostPopular._2))
    }
    
    ssc.start()
    ssc.awaitTermination()
  }
}