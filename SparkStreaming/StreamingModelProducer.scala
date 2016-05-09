package com.lxy.sparkstreaming

import scala.util.Random
import breeze.linalg.DenseVector
import java.net.ServerSocket
import java.io.PrintWriter

/**
 * @author sl169
 * 随机线性回归数据的生成器
 */
object StreamingModelProducer {
  def main(args: Array[String]){
    //每秒处理活动的最大数目
    val MaxEvents = 100
    val NumFeatures = 100
    val random = new Random()
    
    /**
     * 生成服从正态分布的稠密向量的函数
     */
    def generateRandomArray(n: Int) = Array.tabulate(n)(_ => 
    random.nextGaussian())
    
    //生成一个确定的随机模型权重向量
    val w = new DenseVector(generateRandomArray(NumFeatures))
    val intercept = random.nextGaussian() * 10
    
    /**
     * 生成一些随机数据时间
     */
    def generateNoisyData(n: Int) = {
      (1 to n).map{ i => 
        val x = new DenseVector(generateRandomArray(NumFeatures))
        val y: Double = w.dot(x)
        val noisy = y + intercept
        (noisy, x)
      }
    }
    
    //创建网络生成器
    val listener = new ServerSocket(9999)
    println("Listening on port: 9999")
    
    while(true){
      val socket = listener.accept()
      new Thread(){
        override def run = {
          println("Got client connected from: " + socket.getInetAddress)
          val out = new PrintWriter(socket.getOutputStream,true)
          while(true){
            Thread.sleep(1000)
            val num = random.nextInt(MaxEvents)
            val data = generateNoisyData(num)
            data.foreach{ case (y,x) => 
              val xStr = x.data.mkString(",")
              val eventStr = s"$y\t$xStr"
              out.write(eventStr)
              out.write("\n")
            }
            out.flush()
            println(s"Created $num events...")
          }
          socket.close()
        }
      }.start
    }
  }
  
}