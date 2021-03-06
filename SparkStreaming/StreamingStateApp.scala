package com.lxy.sparkstreaming

import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.Seconds

/**
 * @author sl169
 */
object StreamingStateApp {
  def updateState(prices: Seq[(String, Double)], currentTotal: Option[(Int, Double)]) = {
    val currentRevenue = prices.map(_._2).sum
    val currentNumberPurchases = prices.size
    val state = currentTotal.getOrElse((0,0.0))
    Some((currentNumberPurchases + state._1, currentRevenue + state._2))
  }
  
  def main(args: Array[String]){
    val ssc = new StreamingContext("local[2]","FirstStreamingApp",Seconds(10))
    ssc.checkpoint("data/sparkstreaming/")
    val stream = ssc.socketTextStream("localhost", 9999)
    
    val events = stream.map{ record => 
      val event = record.split(",")
      (event(0), event(1), event(2).toDouble)
    }
    
    val users = events.map{case (user, product, price) => (user, (product, price))}
    val revenuePerUser = users.updateStateByKey(updateState)
    revenuePerUser.print()
    
    ssc.start()
    ssc.awaitTermination()
  }
}