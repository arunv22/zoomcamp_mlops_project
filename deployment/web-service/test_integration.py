"""
## Integration Tests for House Price Prediction Service

This module contains integration tests for the house price prediction service, ensuring the end-to-end functionality of the service, from data loading to model training and prediction.

### Overview

- **Module Purpose:** To validate the integration of different components in the house price prediction service.
- **Frameworks Used:** `pytest` for testing, `requests` for HTTP requests, and `flask` for the web service.
- **Key Components Tested:**
  - Data loading and splitting.
  - Model training and logging.
  - Model prediction via a REST API endpoint.

### Code Components

- **Global Constants:**
  - `COLUMN_NAMES`: Names of columns in the dataset.
  - `PREDICT_COLUMNS`: Feature columns used for prediction.
  - `TARGET_COLUMN`: Target column (`MEDV`).
  - `GITHUB_RAW_URL`: URL to fetch the pre-trained model.
  - `RUN_ID`: Identifier for the model run.

- **Functions:**
  - `load_data(url)`: Loads and splits the dataset into training and testing sets.
  - `predict(features)`: Predicts target values using the pre-trained model.
  - `predict_endpoint()`: Flask route handling POST requests for predictions.
  - `home()`: Flask route providing basic information about the service.
  - `run_app()`: Runs the Flask app using the Waitress server.

### Fixtures

- **setup_environment():** Sets up environment variables and configurations.
- **start_server():** Starts the Flask server in a separate thread before tests run, and stops the server after tests are done.

### Tests

- **test_integration_train_log_and_predict():** Integration test to verify the end-to-end functionality of the service:
  1. Loads the dataset.
  2. Trains and logs the model.
  3. Tests the prediction endpoint by sending a sample request and validating the response.

### Notes

- **Dependencies:**
  - `pytest` for running tests.
  - `requests` for HTTP requests.
  - `flask` for creating the web service.
  - `waitress` for serving the Flask app.
  - `numpy`, `pandas`, `sklearn` for data manipulation and machine learning.

- **Model Loading:**
  - The pre-trained model is fetched from a GitHub URL and loaded into memory for predictions.

- **Endpoints:**
  - `/predict`: Handles POST requests to predict house prices.
  - `/`: Provides basic information about the service.

This module ensures that the house price prediction service functions correctly, integrating data loading, model training, logging, and serving predictions via an API.

"""

import os
import pickle
from io import BytesIO
from threading import Thread

import numpy as np
import pandas as pd
import pytest
import requests
from sklearn import linear_model
from flask import Flask, jsonify, request
from sklearn.model_selection import train_test_split
from waitress import serve

from house_price_prediction import ModelManager

# Define column names for data
COLUMN_NAMES = [
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
PREDICT_COLUMNS = ["LSTAT", "INDUS", "NOX", "PTRATIO", "RM", "TAX", "DIS", "AGE"]
TARGET_COLUMN = "MEDV"


def load_data(
    url="https://raw.githubusercontent.com/arunv22/zoomcamp_mlops_project/main/data/housing.csv",
):
    """
    Function to load data from a given URL, split it into training and testing sets,
    and separate features and target variables.

    Parameters:
        url (str): URL to the dataset.

    Returns:
        tuple: (x_train, x_test, y_train, y_test)
    """
    df = pd.read_csv(url, header=None, delimiter=r"\s+", names=COLUMN_NAMES)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    x_train = train_df.loc[:, PREDICT_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df.loc[:, PREDICT_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    return x_train, x_test, y_train, y_test


@pytest.fixture(scope="module")
def setup_environment():
    """
    Setup environment variables, configurations, or any other setup steps.
    This fixture is used to prepare the environment for tests.
    """
    # Potentially other setup code here
    pass


GITHUB_RAW_URL = (
    "https://raw.githubusercontent.com/arunv22/zoomcamp_mlops_project/main/model.pkl"
)
RUN_ID = "ec53532a97e74b2eb5c4a1eeb6834be2"

# Fetch the model file from GitHub and load it into memory
response = requests.get(GITHUB_RAW_URL, timeout=10)
if response.status_code == 200:
    model_file = BytesIO(response.content)
    model = pickle.load(model_file)
    print("Model loaded successfully from GitHub")
else:
    raise Exception(f"Failed to fetch model: {response.status_code} - {response.text}")


def predict(features):
    """
    Function to predict the target value based on input features.

    Parameters:
        features (dict): Input features for prediction.

    Returns:
        float: Predicted target value.
    """
    df = pd.DataFrame([features])
    preds = model.predict(df)
    preds = np.expm1(preds)
    return preds[0]


# Initialize the Flask app
app = Flask("house-price-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Endpoint to handle prediction requests.

    Methods:
        POST: Receives features as JSON, performs prediction, and returns the result.

    Returns:
        JSON: Predicted value and model version.
    """
    features = request.get_json()
    pred = predict(features)

    result = {"MEDV": pred, "model_version": RUN_ID}

    return jsonify(result)


@app.route("/")
def home():
    """
    Home route providing basic information about the service.

    Returns:
        str: Welcome message and instructions for using the prediction service.
    """
    return "Welcome to the house price prediction service. Use POST /predict to get predictions."


# Run Flask app in a separate thread to not block the test
def run_app():
    """
    Function to run the Flask app using Waitress server on a specified port.

    This function is used to start the app server in a separate thread to allow testing.
    """
    port = int(os.environ.get("PORT", 9696))
    serve(app, host="0.0.0.0", port=port)


@pytest.fixture(scope="module", autouse=True)
def start_server():
    """
    Fixture to start the Flask server in a separate thread before tests run,
    and stop the server after tests are done.
    """
    thread = Thread(target=run_app)
    thread.start()
    yield
    thread.join()


@pytest.mark.integration
def test_integration_train_log_and_predict():
    """
    Integration test to verify the end-to-end functionality of training, logging, and predicting.

    Steps:
        1. Load dataset.
        2. Train and log models.
        3. Test the prediction endpoint.
    """
    model_manager = ModelManager()

    # Instantiate and train the model
    model = linear_model.LinearRegression()

    # Load dataset
    x_train, x_test, y_train, y_test = load_data()

    # Train and log models using the updated method
    model_manager.train_model(model, x_train, y_train)

    # Test the prediction endpoint
    url = "http://127.0.0.1:9696/predict"
    headers = {"Content-Type": "application/json"}
    sample = x_test.iloc[0].to_dict()

    response = requests.post(url, json=sample, headers=headers, timeout=10)
    response_data = response.json()

    assert response.status_code == 200
    assert "MEDV" in response_data
    assert "model_version" in response_data
    assert response_data["model_version"] == RUN_ID


if __name__ == "__main__":
    pytest.main()
