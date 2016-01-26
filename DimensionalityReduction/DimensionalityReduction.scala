package com.lxy.ml

import org.apache.spark.SparkContext
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import java.io.File
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import breeze.linalg.DenseMatrix
import breeze.linalg.csvwrite

/**
 * @author sl169
 */
object DimensionalityReduction {
  def main(args: Array[String]){
    val sc = new SparkContext("local", "DimensionalityReduction")
//    val sparkConf = new SparkConf().setAppName("DimensionalityReduction").setMaster("local[*]")
//    val sc = new SparkContext(sparkConf)
//    sc.addJar("DimensionalityReduction.jar")
//    System.setProperty("spark.executor.memory", "2G")
    
//    val path = "facialData/lfw/*"
    
    //在Windows下路径需要这么写：末尾通配符为*\*，lfw目录下所有文件夹下所有文件，而在Linux上则不需要这么写，直接写成lfw/*就行，如上
    val path = "F:\\ScalaWorkSpace\\data\\lfw\\*\\*"
//    val path = "facialData/lfw/*/*"
    val rdd =  sc.wholeTextFiles(path)
//    val first = rdd.first
//    println(first)
    
    //获取文件名，需要将开头的"file:"字符去掉
    val files = rdd.map{ case (fileName, content) => 
      fileName.replace("file:", "") 
    }
//    println(files.first())
    //  /F:/ScalaWorkSpace/data/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg
//    println(files.count)
    //1054
    
//    val aePath = "F:/ScalaWorkSpace/data/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"
//    val aeImage = loadImageFromFile(aePath)
//    println(aeImage)
    
//    val grayImage = processImage(aeImage, 100, 100)
//    println(grayImage)
    
//    ImageIO.write(grayImage, "jpg", new File("F:/ScalaWorkSpace/data/lfw/Aaron_Eckhart/aeGray.jpg"))
    
    val pixels = files.map( f => extractPixels(f, 50, 50))
//    println(pixels.take(10).map(_.take(10).mkString("", ",", ", ...")).mkString("\n"))
    
    //Feature Vector
    val vectors = pixels.map { p => Vectors.dense(p) }
    vectors.setName("image-vectors")
    vectors.cache()
    
    //Normalize the feature vector
    val scaler = new StandardScaler(withMean = true, withStd = false).fit(vectors)
    val scaledVectors = vectors.map(v => scaler.transform(v))
    
    //Running PCA on the LFW dataset
    val matrix = new RowMatrix(scaledVectors)
//    println(matrix.numRows(),matrix.numCols())
    //(1054, 2500)
    val k = 10
    val pc = matrix.computePrincipalComponents(k)
    
    val rows = pc.numRows
    val cols = pc.numCols
//    println(rows, cols)
    //(2500,10)
    
    val pcBreeze = new DenseMatrix(rows, cols, pc.toArray)
    csvwrite(new File("F:/ScalaWorkSpace/data/pc.csv"), pcBreeze)
  }
  
  //Load an image
  def loadImageFromFile(path: String): BufferedImage = {
    ImageIO.read(new File(path))
  }
  
  //Convert an image into a grayscale and resize it into a width*height image
  def processImage(image: BufferedImage, width: Int, height: Int): BufferedImage = {
    val bwImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
    val g = bwImage.getGraphics
    g.drawImage(image, 0, 0, width, height, null)
    g.dispose()
    bwImage
  }
  
  
  //Flatten the width*height image dimensional matrix into a vector, then use this vector as the feature vector 
  def getPixelsFromImage(image: BufferedImage): Array[Double] = {
    val width = image.getWidth
    val height = image.getHeight
    val pixels = Array.ofDim[Double](width * height)
    image.getData.getPixels(0, 0, width, height, pixels)
  }
  
  //Combine the 3 methods above to extract the image's feature
  def extractPixels(path: String, width: Int, height: Int): Array[Double] = {
    val raw = loadImageFromFile(path)
    val processed = processImage(raw, width, height)
    getPixelsFromImage(processed)
  }
}