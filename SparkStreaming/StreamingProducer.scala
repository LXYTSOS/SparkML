package com.lxy.sparkstreaming

import scala.util.Random
import java.net.ServerSocket
import java.io.PrintWriter

/**
 * @author sl169
 */
object StreamingProducer {
  def main(args: Array[String]){
    val random = new Random()
    
    //每秒最大活动数
    val MaxEvents = 6
    
    //读取可能的名称
//    val namesResource = this.getClass.getResourceAsStream("data/names.csv")
    val names = scala.io.Source.fromFile("data/names.csv")
      .getLines()
      .toList
      .head
      .split(",")
      .toSeq
    
    //生成一系列可能的产品
    val products = Seq(
        "iPhone Cover" -> 9.99,
        "Headphones" -> 5.49,
        "Samsung Galaxy Cover" -> 8.95,
        "iPad Cover" -> 7.19)
    
    /**
     * 生成随机产品活动
     */
    def generateProductEvents(n: Int) = {
    	(1 to n).map{ i => 
    	val (product, price) = products(random.nextInt(products.size))
      val user = random.shuffle(names).head
      (user,product,price)}
    }
    
    //创建网络生成器
    val listener = new ServerSocket(9999)
    println("Listening on port: 9999")
    
    while(true){
      val socket = listener.accept()
      new Thread(){
        override def run = {
          println("Got client connected from: "+socket.getInetAddress)
          val out = new PrintWriter(socket.getOutputStream,true)
          
          while(true){
            Thread.sleep(1000)
            val num = random.nextInt(MaxEvents)
            val productEvents = generateProductEvents(num)
            productEvents.foreach{ event => 
              out.write(event.productIterator.mkString(","))
              out.write("\n")
            }
            out.flush()
            println(s"Created $num events...")
          }
          socket.close()
        }
      }.start()
    }
  }
  
}