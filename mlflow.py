import copy
import json

import mlforecast
import pandas as pd
import polars as pl
from mlforecast import MLForecast
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import bias, mae, rmse

import mlflow
from config import config
from src.utils import CompetitionMetric

# Load preprocessed dataset and convert to pandas
df = (
    pl.read_parquet("data/preprocessed/sales.parquet")
    .to_pandas()
    .assign(
        Client=lambda df: df["Client"].astype("int"),
        Product=lambda df: df["Product"].astype("int"),
        Warehouse=lambda df: df["Warehouse"].astype("int"),
    )
)

# Extract configuration details
static_features = config["fit"]["static_features"]
reference_features = config["reference_features"]

# Initialize the forecast model
mlf = MLForecast(**config["init"])

# Set up MLflow experiment
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("vn1-forecasting")

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

    # Log model
    mlforecast.flavor.log_model(model=mlf, artifact_path="model")
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

# Load model
loaded_model = mlforecast.flavor.load_model(model_uri=model_uri)
results = loaded_model.predict(h=3, X_df=df[static_features + reference_features], ids=[3])
print(results.head(2))

# PyFunc
loaded_pyfunc = mlforecast.flavor.pyfunc.load_model(model_uri=model_uri)
predict_conf = pd.DataFrame(
    [
        {
            "h": 3,
            "ids": [0, 2],
            "X_df": df[static_features + reference_features],
            "level": [80],
        }
    ]
)
pyfunc_result = loaded_pyfunc.predict(predict_conf)
print(pyfunc_result.head(2))
