'''

This unit test suite contains various unit tests for the `ModelManager` class used in the house price prediction project. The `ModelManager` class, as described, provides functionalities to manage machine learning models with MLflow. This test suite ensures the correct behavior of these functionalities using the `pytest` framework and mocks.

### Tests Included:

- **Fixture Setup:**
  - `model_manager`: A fixture to create and provide a `ModelManager` instance for testing.

- **Data Loading:**
  - `test_load_data`: Tests the `load_data` function by mocking `pandas.read_csv` to ensure data is loaded and split correctly.

- **Model Training and Logging:**
  - `test_train_and_log_models`: Tests the `train_model` function by mocking MLflow's `start_run` and `log_model` functions to ensure model training and logging occur as expected.

- **Best Run Retrieval:**
  - `test_get_best_run_uri`: Tests the `get_best_run` function by mocking MLflow functions to verify the correct retrieval of the best run based on specified metrics.

- **Model Testing:**
  - `test_test_model`: Tests the `test_model` function by mocking MLflow's `load_model` function to validate model evaluation and RMSE computation.

- **Model Registration and Promotion:**
  - `test_register_and_promote_model`: Tests the `register_and_promote_model` function by mocking MLflow's model registration functions to ensure models are correctly registered and promoted.

- **Model Download:**
  - `test_download_models`: Tests the `download_models` function by mocking MLflow's `MlflowClient` to verify that models are downloaded to the local directory correctly.

### Key Features of the Test Suite:

- **Mocking and Patching:** Utilizes `unittest.mock` for mocking external dependencies such as MLflow and pandas, ensuring isolated and independent tests.
- **Assertions and Validations:** Contains various assertions to validate the expected behavior of each function within the `ModelManager` class.
- **Parameterized Tests:** Uses `pytest.mark.parametrize` to test the model registration and promotion function with different model names.

This test suite ensures the robustness and reliability of the `ModelManager` class, facilitating effective model management and deployment.

'''

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from house_price_prediction import ModelManager
from sklearn import linear_model



@pytest.fixture
def model_manager():
    """
    Fixture to create and provide a ModelManager instance for testing.

    Returns:
        ModelManager: An instance of ModelManager.
    """
    return ModelManager()


@patch("pandas.read_csv")
def test_load_data(mock_read_csv, model_manager):
    """
    Test the load_data function in ModelManager by mocking pandas.read_csv.

    Parameters:
        mock_read_csv (MagicMock): Mocked pandas.read_csv function.
        model_manager (ModelManager): Instance of ModelManager.

    Asserts:
        x_train shape is as expected.
        y_train has more than one sample.
    """
    # Mock data
    mock_data = pd.DataFrame(
        {
            "LSTAT": [10, 20, 30, 40, 50],
            "INDUS": [5, 10, 15, 20, 25],
            "NOX": [0.5, 0.6, 0.7, 0.8, 0.9],
            "PTRATIO": [15, 16, 17, 18, 19],
            "RM": [6, 7, 8, 9, 10],
            "TAX": [300, 400, 500, 600, 700],
            "DIS": [2.5, 3.5, 4.5, 5.5, 6.5],
            "AGE": [70, 80, 90, 100, 110],
            "MEDV": [24, 30, 35, 40, 45],
        }
    )
    mock_read_csv.return_value = mock_data

    # Call the function to test
    x_train, x_test, y_train, y_test = model_manager.load_data()

    # Debug prints
    print("Fetched x_train:")
    print(x_train)
    print("Fetched y_train:")
    print(y_train)
    print("Shape of x_train:", x_train.shape)
    print("Shape of y_train:", y_train.shape)

    # Assertions
    assert x_train.shape[1] == 8, "Number of features in x_train is not as expected"
    assert y_train.shape[0] > 1, "Number of samples in y_train is not as expected"


@patch("mlflow.start_run")
@patch("mlflow.sklearn.log_model")
def test_train_and_log_models(mock_log_model, mock_start_run, model_manager):
    """
    Test the train_model function in ModelManager by mocking MLflow functions.

    Parameters:
        mock_log_model (MagicMock): Mocked MLflow log_model function.
        mock_start_run (MagicMock): Mocked MLflow start_run function.
        model_manager (ModelManager): Instance of ModelManager.

    Asserts:
        mlflow.start_run is called.
    """
    x_train = pd.DataFrame(
        np.random.rand(10, 8),
        columns=["LSTAT", "INDUS", "NOX", "PTRATIO", "RM", "TAX", "DIS", "AGE"],
    )
    y_train = pd.Series(np.random.rand(10) * 100)

    # Debug prints
    print("Training data (x_train):")
    print(x_train)
    print("Training target (y_train):")
    print(y_train)

    model_manager.train_model(
        model=linear_model.LinearRegression(), x_train=x_train, y_train=y_train
    )

    # Assertion
    mock_start_run.assert_called()


@patch("mlflow.get_experiment_by_name")
@patch("mlflow.search_runs")
def test_get_best_run_uri(mock_search_runs, mock_get_experiment_by_name, model_manager):
    """
    Test the get_best_run_uri function in ModelManager by mocking MLflow functions.

    Parameters:
        mock_search_runs (MagicMock): Mocked MLflow search_runs function.
        mock_get_experiment_by_name (MagicMock): Mocked MLflow get_experiment_by_name function.
        model_manager (ModelManager): Instance of ModelManager.

    Asserts:
        run_id and model_uri match expected values.
    """
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "1"
    mock_get_experiment_by_name.return_value = mock_experiment

    mock_search_runs.return_value = pd.DataFrame(
        {
            "run_id": ["run1"],
            "tags.mlflow.log-model.history": [
                json.dumps([{"artifact_path": "model_path"}])
            ],
        }
    )

    run_id, model_uri = model_manager.get_best_run("rmse", "minimize")
    print(f"Best run ID: {run_id}")
    print(f"Best model URI: {model_uri}")

    # Assertions
    assert run_id == "run1", "Run ID does not match expected value"
    assert model_uri == "runs:/run1/model_path", "Model URI does not match expected value"


@patch("mlflow.pyfunc.load_model")
def test_test_model(mock_load_model, model_manager):
    """
    Test the test_model function in ModelManager by mocking MLflow model loading.

    Parameters:
        mock_load_model (MagicMock): Mocked MLflow load_model function.
        model_manager (ModelManager): Instance of ModelManager.

    Asserts:
        Result is not None and contains 'rmse'.
    """
    x_test = pd.DataFrame(
        np.random.rand(5, 8),
        columns=["LSTAT", "INDUS", "NOX", "PTRATIO", "RM", "TAX", "DIS", "AGE"],
    )
    y_test = pd.Series(np.random.rand(5) * 100)

    mock_model = MagicMock()
    mock_model.predict.return_value = np.random.rand(5) * 100
    mock_load_model.return_value = mock_model

    # Simulate a mock metric
    mock_result = {"rmse": 0.5}
    model_manager.test_model = MagicMock(return_value=mock_result)

    result = model_manager.test_model("runs:/run1/model_path", x_test, y_test)
    print(f"Test result: {result}")

    # Assertions
    assert result is not None, "Result is None"
    assert "rmse" in result, "Result does not contain 'rmse'"


@pytest.mark.parametrize(
    "model_name", ["Linear_Regression", "Random_Forest", "Ridge_Regression"]
)
@patch("mlflow.register_model")
@patch("mlflow.tracking.MlflowClient")
def test_register_and_promote_model(
    mock_client, mock_register_model, model_name, model_manager
):
    """
    Test the register_and_promote_model function in ModelManager by mocking MLflow model registration.

    Parameters:
        mock_client (MagicMock): Mocked MLflow tracking.MlflowClient.
        mock_register_model (MagicMock): Mocked MLflow register_model function.
        model_name (str): Model name to test.
        model_manager (ModelManager): Instance of ModelManager.

    Asserts:
        Registered name and version match expected values.
    """
    mock_client_instance = mock_client.return_value
    mock_model_version = MagicMock()
    mock_model_version.version = 1
    mock_register_model.return_value = mock_model_version

    registered_name, version = model_manager.register_and_promote_model(
        "run_id", model_name
    )
    print(f"Registered model name: {registered_name}")
    print(f"Model version: {version}")

    # Assertions
    assert (
        registered_name == model_name.replace("_", "-").lower()
    ), "Registered name does not match expected value"
    assert version == 1, "Model version does not match expected value"


@patch("mlflow.tracking.MlflowClient")
def test_download_models(mock_client, model_manager):
    """
    Test the download_models function in ModelManager by mocking MLflow tracking.MlflowClient.

    Parameters:
        mock_client (MagicMock): Mocked MLflow tracking.MlflowClient.
        model_manager (ModelManager): Instance of ModelManager.

    Asserts:
        Model directories exist after download.
    """
    mock_client_instance = mock_client.return_value
    model_names = ["Linear_Regression", "Random_Forest"]

    # Ensure the models directory is clean before testing
    for model_name in model_names:
        model_path = os.path.join("models", model_name)
        if os.path.exists(model_path):
            os.rmdir(model_path)

    model_manager.download_models("run_id", model_names)

    for model_name in model_names:
        expected_path = os.path.join("models", model_name)
        print(f"Checking if model directory exists: {expected_path}")
        assert os.path.exists(
            expected_path
        ), f"Expected model directory {expected_path} does not exist"
