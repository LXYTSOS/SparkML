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
import breeze.linalg.DenseVector

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
//    val k = 10
    //这个步骤比较耗时（大概5分钟），比SVD（1分钟左右）要慢许多
//    val pc = matrix.computePrincipalComponents(k)
    
//    val rows = pc.numRows
//    val cols = pc.numCols
//    println(rows, cols)
    //(2500,10)
    
    //Save principle components as csv file
//    val pcBreeze = new DenseMatrix(rows, cols, pc.toArray)
//    csvwrite(new File("F:/ScalaWorkSpace/data/pc.csv"), pcBreeze)
    
    //Projecting data using PCA on the FLW dataset
//    val projected = matrix.multiply(pc)
//    println(projected.numRows(),projected.numCols())
    //(1054,10)
//    println(projected.rows.take(5).mkString("\n"))
    
//    val svd = matrix.computeSVD(10, computeU = true)
//    println(s"U dimension: (${svd.U.numRows}, ${svd.U.numCols})")
//    println(s"S dimension: (${svd.s.size}, )")
//    println(s"V dimension: (${svd.V.numRows}, ${svd.V.numCols})")
    // U dimension: (1054, 10)
    // S dimension: (10, )
    // V dimension: (2500, 10)
    
//    println(approxEqual(svd.V.toArray, pc.toArray))
    
    //Compare U and S matrix with PCA projection
//    val breezeS = DenseVector(svd.s.toArray)
//    val projectedSVD = svd.U.rows.map{ v => 
//      val breezeV = DenseVector(v.toArray)
//      val multV = breezeV :* breezeS
//      Vectors.dense(multV.data)
//    }
//    val result = projected.rows.zip(projectedSVD).map{ case (v1, v2) => 
//      approxEqual(v1.toArray, v2.toArray)}.filter(b => true).count
//    println(result)
    //1054
    
    //Evaluating k for SVD on the LFW dataset
//    val sValues = (1 to 5).map{ i => matrix.computeSVD(i, computeU = false).s}
//    sValues.foreach(println)
    
    //计算前300个奇异值，并保存为csv文件，使用Python绘图观察图像从哪个点开始变平缓
    val svd300 = matrix.computeSVD(300, computeU = false)
    val sMatrix = new DenseMatrix(1,300, svd300.s.toArray)
    csvwrite(new File("F:/ScalaWorkSpace/data/s300.csv"), sMatrix)
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
  
  //Compare V matrix from svd with the result of PCA
  def approxEqual(array1: Array[Double], array2: Array[Double], tolerance: Double = 1e-6): Boolean = {
    val bools = array1.zip(array2).map{ case (v1, v2) => if
      (math.abs(math.abs(v1) - math.abs(v2)) > 1e-6) false else true}
    bools.fold(true)(_&_)
  }
  
  
}