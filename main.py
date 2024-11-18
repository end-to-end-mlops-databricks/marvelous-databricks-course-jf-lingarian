from src.preprocessor import DataProcessor

DataProcessor("data/raw", "sales").process_files()
DataProcessor("data/raw", "price").process_files()
