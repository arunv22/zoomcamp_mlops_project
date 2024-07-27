"""
This module contains the ModelManager class, which is used for managing machine learning
models using MLflow for house price prediction.
"""

import json
import os

import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler


class ModelManager:
    """Initializes the ModelManager class with MLflow tracking URI and experiment name."""

    def __init__(
        self,
        tracking_uri="http://localhost:5000",
        experiment_name="house-price-prediction",
    ):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self._restore_or_active_experiment()

    def _restore_or_active_experiment(self):
        """Restores the experiment if it is deleted, otherwise ensures it is active."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            raise ValueError("Experiment not found.")
        if experiment.lifecycle_stage == "deleted":
            client = mlflow.tracking.MlflowClient()
            client.restore_experiment(experiment.experiment_id)
            print(f"Experiment '{self.experiment_name}' has been restored.")
        else:
            print(f"Experiment '{self.experiment_name}' is already active.")

    @staticmethod
    def load_data(
        url="https://raw.githubusercontent.com/arunv22/zoomcamp_mlops_project/main/data/housing.csv",
    ):
        """Loads the dataset from a URL and splits it into training and testing sets."""
        column_names = [
            "CRIM",
            "ZN",
            "INDUS",
            "CHAS",
            "NOX",
            "RM",
            "AGE",
            "DIS",
            "RAD",
            "TAX",
            "PTRATIO",
            "B",
            "LSTAT",
            "MEDV",
        ]
        df = pd.read_csv(url, header=None, delimiter=r"\s+", names=column_names)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        x_train = train_df.loc[
            :, ["LSTAT", "INDUS", "NOX", "PTRATIO", "RM", "TAX", "DIS", "AGE"]
        ]
        y_train = train_df["MEDV"]
        x_test = test_df.loc[
            :, ["LSTAT", "INDUS", "NOX", "PTRATIO", "RM", "TAX", "DIS", "AGE"]
        ]
        y_test = test_df["MEDV"]

        return x_train, x_test, y_train, y_test

    @staticmethod
    def create_pipeline(model):
        """Creates a preprocessing and model training pipeline."""
        return Pipeline(
            [
                (
                    "preprocessor",
                    ColumnTransformer(
                        transformers=[
                            (
                                "log",
                                FunctionTransformer(func=np.log1p, validate=False),
                                [
                                    "LSTAT",
                                    "INDUS",
                                    "NOX",
                                    "PTRATIO",
                                    "RM",
                                    "TAX",
                                    "DIS",
                                    "AGE",
                                ],
                            ),
                            (
                                "scaler",
                                MinMaxScaler(),
                                [
                                    "LSTAT",
                                    "INDUS",
                                    "NOX",
                                    "PTRATIO",
                                    "RM",
                                    "TAX",
                                    "DIS",
                                    "AGE",
                                ],
                            ),
                        ],
                        remainder="passthrough",
                    ),
                ),
                ("model", model),
            ]
        )

    def train_model(self, model, x_train, y_train):
        """Trains the model and logs the training run with MLflow."""
        pipeline = self.create_pipeline(model)

        with mlflow.start_run() as run:
            pipeline.fit(x_train, y_train)

            y_pred = pipeline.predict(x_train)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))
            print(f"Training RMSE: {rmse:.2f}")

            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            mlflow.log_metric("rmse", rmse)

    def get_best_run(self, metric_name="rmse", metric_goal="minimize"):
        """Finds the best run based on the specified metric and returns its details."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            raise ValueError("Experiment not found.")

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            order_by=[
                f"metrics.{metric_name} {'asc' if metric_goal == 'minimize' else 'desc'}"
            ],
        )

        if not runs.empty:
            best_run = runs.iloc[0]
            best_run_id = best_run["run_id"]
            best_artifact_path = json.loads(best_run["tags.mlflow.log-model.history"])[
                0
            ]["artifact_path"]

            return best_run_id, f"runs:/{best_run_id}/{best_artifact_path}"
        else:
            raise ValueError("No runs found in the experiment.")


    def test_model(self, logged_model_uri, x_test, y_test):
        """Tests a model loaded from the specified URI and prints the RMSE."""
        try:
            loaded_model = mlflow.pyfunc.load_model(logged_model_uri)
            y_pred = loaded_model.predict(pd.DataFrame(x_test))
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"RMSE for best model selected: {rmse:.2f}")

            return {"rmse": rmse}
        except MlflowException as e:
            print(f"Error loading model from {logged_model_uri}: {e}")

    @staticmethod
    def register_and_promote_model(run_id, model_name, stage="Production"):
        """Registers a model and promotes it to the specified stage."""
        model_uri = f"runs:/{run_id}/models/{model_name}"
        registered_model_name = model_name.replace("_", "-").lower()

        model_version = mlflow.register_model(model_uri, registered_model_name)
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=registered_model_name, version=model_version.version, stage=stage
        )

        return registered_model_name, model_version.version

    @staticmethod
    def download_models(run_id, model_names, output_dir="models"):
        """Downloads models from MLflow to a local directory."""
        os.makedirs(output_dir, exist_ok=True)
        client = mlflow.tracking.MlflowClient()

        for model_name in model_names:
            artifact_path = f"models/{model_name}/"
            local_path = os.path.join(output_dir, model_name)
            os.makedirs(local_path, exist_ok=True)
            client.download_artifacts(run_id, artifact_path, local_path)
            print(f"Downloaded {model_name} to local path: {local_path}")
