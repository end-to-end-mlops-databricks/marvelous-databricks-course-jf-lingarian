# Databricks notebook source
import json
import os

import mlflow

# Set up MLflow tracking URI and experiment
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_name="/Shared/vn1-forecasting")
mlflow.set_experiment_tags({"repository_name": "vn1-forecasting"})

# COMMAND ----------

# Search for experiments with the specified tag
experiments = mlflow.search_experiments(filter_string="tags.repository_name='vn1-forecasting'")
print(experiments)

# COMMAND ----------

# Save experiment details to a JSON file
with open("mlflow_experiment.json", "w") as json_file:
    json.dump(experiments[0].__dict__, json_file, indent=4)

# Start an MLflow run and log parameters and metrics
with mlflow.start_run(
    run_name="demo-run",
    tags={"git_sha": "ffa63b430205ff7", "branch": "week2"},
    description="demo run",
) as run:
    mlflow.log_params({"type": "demo"})
    mlflow.log_metrics({"metric1": 1.0, "metric2": 2.0})

# Retrieve the run ID and run information
run_id = mlflow.search_runs(
    experiment_names=["/Shared/vn1-forecasting"],
    filter_string="tags.git_sha='ffa63b430205ff7'",
).run_id[0]
run_info = mlflow.get_run(run_id=f"{run_id}").to_dictionary()
print(run_info)

# Save run information to a JSON file
with open("run_info.json", "w") as json_file:
    json.dump(run_info, json_file, indent=4)

# Print metrics and parameters from the run information
print(run_info["data"]["metrics"])
print(run_info["data"]["params"])
