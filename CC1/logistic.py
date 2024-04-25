from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidator

import logging
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Initialize Spark Session
spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

try:
    # Load data
    data_path = r"./TrainingDataset.csv"
    data = spark.read.csv(data_path, sep=';', header=True, inferSchema=True)
    logging.info("Data loaded successfully.")

    # Clean up column names by removing extra quotes
    for col_name in data.columns:
        data = data.withColumnRenamed(col_name, col_name.replace('"', ''))
    logging.info("Column names cleaned.")

    # Assemble features
    feature_columns = data.columns[:-1]  # all columns except the last one 'quality'
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    
    # Scale features
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

    # Define the Logistic Regression model
    lr = LogisticRegression(featuresCol='scaledFeatures', labelCol='quality')

    # Create a pipeline
    pipeline = Pipeline(stages=[assembler, scaler, lr])

    # Define ParamGrid for Cross Validation
    paramGrid = (ParamGridBuilder()
                 .addGrid(lr.regParam, [0.01, 0.1, 0.5])
                 .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
                 .addGrid(lr.maxIter, [10, 50, 100])
                 .build())

    # Define evaluator
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")

    # Set up 3-fold cross-validation
    crossVal = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=3)

    cvModel = crossVal.fit(data)
    logging.info("Model trained.")

    cvModel.bestModel.write().overwrite().save("./BestWineQualityPipelineModel")
    logging.info("Best PipelineModel saved successfully.")
    # After fitting the cross-validator

# Get the best model
    best_model = cvModel.bestModel

# Get the index of the best model's F1 score
    best_f1_index = cvModel.avgMetrics.index(max(cvModel.avgMetrics))

# Retrieve the F1 score of the best model
    best_f1_score = cvModel.avgMetrics[best_f1_index]

    logging.info(f"Best F1 Score: {best_f1_score}")

except Exception as e:
    logging.error("An error occurred:", exc_info=True)

finally:
    # Stop Spark session
    spark.stop()
    logging.info("Spark session stopped.")
