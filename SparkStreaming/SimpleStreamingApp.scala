package com.lxy.sparkstreaming

import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.Seconds
import org.apache.log4j.Logger

/**
 * @author sl169
 */
object SimpleStreamingApp {
  val logger = Logger.getLogger("Streaming")
  def main(args: Array[String]){
    val ssc = new StreamingContext("local[2]","First Streaming App", Seconds(10))
    val stream = ssc.socketTextStream("localhost", 9999)
    
    logger.info(stream.print())
    ssc.start()
    ssc.awaitTermination()
  }
}