import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
val sqlContext = spark.sqlContext
var data = spark.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("/home/docubeapp-usr/export.csv")
data.createOrReplaceTempView("data")
val labelIndexer = new StringIndexer().setInputCol("State").setOutputCol("indexed"+"State").fit(data)
data=labelIndexer.transform(data)
data.createOrReplaceTempView("data")
var df = sqlContext.sql("select  `2014 Population estimate` as estimate,`2015 median sales price` as label,indexedState from data")
df.createOrReplaceTempView("df")
val assembler = new VectorAssembler().setInputCols(Array("estimate","indexedState")).setOutputCol("features")
df = assembler.transform(df)
df.createOrReplaceTempView("df")
var df = sqlContext.sql("select   features,  label from df")
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(df)
val rf = new RandomForestRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures").setMaxBins(43)
val pipeline = new Pipeline().setStages(Array(featureIndexer, rf))
val model = pipeline.fit(df)
var predictions = model.transform(df)
val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)


