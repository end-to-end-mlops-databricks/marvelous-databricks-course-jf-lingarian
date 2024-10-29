from src.preprocessor import DataProcessor
import pandas as pd

DataProcessor("data/raw", "sales").process_files()
DataProcessor("data/raw", "price").process_files()