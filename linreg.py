from pyspark.ml.regression import LinearRegression, LinearRegressionModel
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

linreg = LinearRegression(maxIter=10, regParam=0.001, elasticNetParam=0.0, labelCol="quality")

if sys.argv[1] == 'train':
    linregmodel = linreg.fit(traindata)
    linregmodel.write().overwrite().save(os.path.join(pathlib.Path().resolve(), 'linregmodel'))

if sys.argv[1] == 'test':
    linregmodel = LinearRegressionModel.load(os.path.join(pathlib.Path().resolve(), 'linregmodel'))

    prediction = linregmodel.transform(testdata)
    prediction.select("prediction", "quality", "features").show(10)

    results = linregmodel.evaluate(testdata)
    print("RMSE: %f" % results.rootMeanSquaredError)

spark.stop()