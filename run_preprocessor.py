from src.preprocessor import DataProcessor
from config import config_project
from pyspark.sql import SparkSession

print(config_project)
# Initialize data preprocessor
sales_processor = DataProcessor(config_project['sales_path'], 'sales', config_project)

# Initialize spark session
spark = SparkSession.builder.getOrCreate()

# Preprocess and save sales
sales_processor.process_files()
sales_processor.save_to_catalog(spark)

if config_project['use_price']:
    price_processor = DataProcessor(config_project["price_path"], 'sales', config_project)