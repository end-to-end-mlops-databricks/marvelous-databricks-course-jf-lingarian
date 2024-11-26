# Databricks notebook source
import copy
import json

import mlforecast.flavor
import mlforecast
import pandas as pd
import polars as pl
from mlforecast import MLForecast
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import bias, mae, rmse

import mlflow
from mlflow.models import infer_signature
from config import config
from src.utils import CompetitionMetric
from pyspark.sql import SparkSession
from config import config_project
from mlflow import MlflowClient
from mlflow.utils.environment import _mlflow_conda_env
from mlflow import MlflowClient

# Initialize mlflow 
client = MlflowClient()

# Extract configuration parameters
catalog_name = config_project['catalog']
schema_name = config_project['schema']

# Initialize spark session
spark = SparkSession.builder.getOrCreate()

# Load preprocessed dataset and convert to pandas
df_spark = spark.table(f"{catalog_name}.{schema_name}.processed_data_sales")
df = df_spark.toPandas().assign(ds=lambda df: pd.to_datetime(df['ds']))
df

# COMMAND ----------

# Extract configuration details
static_features = config["fit"]["static_features"]
reference_features = config["reference_features"]

# Convert object cols into integers
for col in config['fit']['static_features']:
  df[col] = df[col].astype(int)

# Initialize the forecast model
mlf = MLForecast(**config["init"])

# Set up MLflow experiment
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_name="/Shared/vn1-forecasting")
git_sha = "ffa63b430205ff7"

with mlflow.start_run() as run:
    # Perform cross validation
    cv_df = mlf.cross_validation(df[static_features + reference_features], **config["fit"])

    # Aggregate metrics from cross validation results
    cv_evaluate = evaluate(
        cv_df.drop(columns="cutoff"),
        metrics=[rmse, mae, CompetitionMetric, bias],
        agg_fn="mean",
    )

    # Create a dictionary with aggregated evaluation metrics
    dict_metrics = {k: float(i) for k, i in cv_evaluate.set_index("metric")["lgb"].items()}

    # Log parameters
    logged_params = copy.deepcopy(config)
    logged_params["init"]["models"] = {
        k: (v.__class__.__name__, v.get_params()) for k, v in config["init"]["models"].items()
    }
    mlflow.log_params(logged_params)

    # Log metrics
    mlflow.log_metrics(dict_metrics)

    # Fit model to entire dataset
    mlf.fit(df[static_features + reference_features], **{key: config["fit"][key] for key in list(config["fit"].keys())[:-2]})

    # create preditions
    y_pred = mlf.predict(h=3)

    # Infer signature
    signature = infer_signature(None, y_pred)

    # Log model
    mlforecast.flavor.log_model(model=mlf, artifact_path="model", signature=signature)
    model_uri = mlflow.get_artifact_uri("model")
    run_id = run.info.run_id

# Save experiment details to a JSON file
experiments = mlflow.search_experiments(filter_string="tags.repository_name='vn1-forecasting'")
with open("mlflow_experiment.json", "w") as json_file:
    json.dump(experiments[0].__dict__, json_file, indent=4)

# Retrieve the run ID and run information
run_info = mlflow.get_run(run_id=f"{run_id}").to_dictionary()
with open("run_info.json", "w") as json_file:
    json.dump(run_info, json_file, indent=4)

# Print metrics and parameters from the run information
print(run_info["data"]["metrics"])
print(run_info["data"]["params"])

# COMMAND ----------

run_info

# COMMAND ----------

model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/model',
    name=f"{catalog_name}.{schema_name}.forecasting_model",
    tags={"git_sha": git_sha})

# COMMAND ----------

with open("model_version.json", "w") as json_file:
    json.dump(model_version.__dict__, json_file, indent=4)

# COMMAND ----------

model_name = f"{catalog_name}.{schema_name}.forecasting_model"

model_version_alias = "test"
client.set_registered_model_alias(model_name, model_version_alias, "1")  
 
model_uri = f"models:/{model_name}@{model_version_alias}"
model = mlforecast.flavor.load_model(model_uri=model_uri)
model

# COMMAND ----------

model.predict(h=3)

# COMMAND ----------

# Load model
loaded_model = mlforecast.flavor.load_model(model_uri=model_uri)
results = loaded_model.predict(h=3, ids=['38/63/11261','9/82/9950'])
print(results.head(2))

# PyFunc
loaded_pyfunc = mlforecast.flavor.pyfunc.load_model(model_uri=model_uri)
predict_conf = pd.DataFrame(
    [
        {
            "h": 3,
            "ids": ['38/63/11261','9/82/9950'],
        }
    ]
)
pyfunc_result = loaded_pyfunc.predict(predict_conf)
print(pyfunc_result.head(2))
