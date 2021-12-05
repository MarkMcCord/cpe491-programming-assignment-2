from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

import sys
import pathlib
import os

spark = SparkSession\
    .builder\
    .appName("Lin Reg")\
    .getOrCreate()

input = spark.read.option("delimiter", ";").option("header", "true").option("inferSchema", True).csv("winequality-white.csv")

traininput, testinput = input.randomSplit([.8, .2], 1234567)

assembler = VectorAssembler(inputCols=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"], outputCol="features")
traindata = assembler.transform(traininput)
testdata = assembler.transform(testinput)

ranforreg = RandomForestRegressor(featuresCol="features", labelCol="quality", numTrees=50, maxDepth=25)

if sys.argv[1] == 'train':
    ranforregmodel = ranforreg.fit(traindata)
    ranforregmodel.write().overwrite().save(os.path.join(pathlib.Path().resolve(), 'ranforregmodel'))

if sys.argv[1] == 'test':
    ranforregmodel = RandomForestRegressionModel.load(os.path.join(pathlib.Path().resolve(), 'ranforregmodel'))

    prediction = ranforregmodel.transform(testdata)
    prediction.select("prediction", "quality", "features").show(10)

    results = RegressionEvaluator(labelCol="quality", predictionCol="prediction", metricName="rmse")
    print("RMSE: %f" % results.evaluate(prediction))

spark.stop()