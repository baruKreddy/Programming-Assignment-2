import logging
import argparse
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Setup command line arguments
parser = argparse.ArgumentParser(description='Predict Wine Quality')
parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file for prediction')
args = parser.parse_args()

# Initialize Spark Session
spark = SparkSession.builder.appName("WineQualityPrediction-Predict").getOrCreate()

try:
    # Load the pre-trained model
    model_path = "./BestWineQualityPipelineModel"
    model = PipelineModel.load(model_path)
    logging.info("Model loaded successfully.")

    # Load the data for prediction
    predict_data_path = args.data_path  # Use the command line argument
    predict_data = spark.read.csv(predict_data_path, sep=';', header=True, inferSchema=True)
    logging.info(f"Prediction data loaded successfully from {predict_data_path}.")

    # Clean up column names by removing extra quotes
    for col_name in predict_data.columns:
        predict_data = predict_data.withColumnRenamed(col_name, col_name.replace('"', ''))
    logging.info("Column names cleaned.")

    # Check if 'features' column already exists, if so drop it
    if "features" in predict_data.columns:
        predict_data = predict_data.drop("features")

    # Apply model to make predictions
    predictions = model.transform(predict_data)
    logging.info("Predictions made successfully.")

    # Setup the evaluator
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")

    # Calculate F1 score
    f1_score = evaluator.evaluate(predictions)
    logging.info(f"F1 Score: {f1_score}")

except Exception as e:
    logging.error("An error occurred:", exc_info=True)

finally:
    # Stop Spark session
    spark.stop()
    logging.info("Spark session stopped.")
