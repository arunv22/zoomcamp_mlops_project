import os
import requests
import pickle
import pandas as pd
import numpy as np
from io import BytesIO
from flask import Flask, request, jsonify
from house_price_prediction import ModelManager
import pytest
from threading import Thread
from waitress import serve
from sklearn.model_selection import train_test_split

# Define column names for data
COLUMN_NAMES = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
PREDICT_COLUMNS = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
TARGET_COLUMN = 'MEDV'


def load_data(url="https://raw.githubusercontent.com/arunv22/zoomcamp_mlops_project/main/data/housing.csv"):
    '''
    Function to load data from a given URL, split it into training and testing sets,
    and separate features and target variables.

    Parameters:
        url (str): URL to the dataset.

    Returns:
        tuple: (x_train, x_test, y_train, y_test)
    '''
    df = pd.read_csv(url, header=None, delimiter=r"\s+", names=COLUMN_NAMES)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    x_train = train_df.loc[:, PREDICT_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df.loc[:, PREDICT_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    return x_train, x_test, y_train, y_test


@pytest.fixture(scope="module")
def setup_environment():
    '''
    Setup environment variables, configurations, or any other setup steps.
    This fixture is used to prepare the environment for tests.
    '''
    # Pass column names to be used in tests or other setup if needed
    global PREDICT_COLUMNS, TARGET_COLUMN
    # Potentially other setup code here
    pass


GITHUB_RAW_URL = "https://raw.githubusercontent.com/arunv22/zoomcamp_mlops_project/main/model.pkl"
RUN_ID = 'ec53532a97e74b2eb5c4a1eeb6834be2'

# Fetch the model file from GitHub and load it into memory
response = requests.get(GITHUB_RAW_URL)
if response.status_code == 200:
    model_file = BytesIO(response.content)
    model = pickle.load(model_file)
    print("Model loaded successfully from GitHub")
else:
    raise Exception(f"Failed to fetch model: {response.status_code} - {response.text}")


def predict(features):
    '''
    Function to predict the target value based on input features.

    Parameters:
        features (dict): Input features for prediction.

    Returns:
        float: Predicted target value.
    '''
    df = pd.DataFrame([features])
    preds = model.predict(df)
    preds = np.expm1(preds)
    return preds[0]


# Initialize the Flask app
app = Flask('house-price-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    '''
    Endpoint to handle prediction requests.

    Methods:
        POST: Receives features as JSON, performs prediction, and returns the result.

    Returns:
        JSON: Predicted value and model version.
    '''
    features = request.get_json()
    pred = predict(features)

    result = {
        'MEDV': pred,
        'model_version': RUN_ID  # You can update this if needed
    }

    return jsonify(result)


@app.route('/')
def home():
    '''
    Home route providing basic information about the service.

    Returns:
        str: Welcome message and instructions for using the prediction service.
    '''
    return "Welcome to the house price prediction service. Use POST /predict to get predictions."


# Run Flask app in a separate thread to not block the test
def run_app():
    '''
    Function to run the Flask app using Waitress server on a specified port.

    This function is used to start the app server in a separate thread to allow testing.
    '''
    port = int(os.environ.get('PORT', 9696))
    serve(app, host='0.0.0.0', port=port)


@pytest.fixture(scope="module", autouse=True)
def start_server():
    '''
    Fixture to start the Flask server in a separate thread before tests run,
    and stop the server after tests are done.
    '''
    thread = Thread(target=run_app)
    thread.start()
    yield
    thread.join()

@pytest.mark.integration
def test_integration_train_log_and_predict(setup_environment):
    '''
    Integration test to verify the end-to-end functionality of training, logging, and predicting.

    Steps:
        1. Load dataset.
        2. Train and log models.
        3. Test the prediction endpoint.
    '''
    model_manager = ModelManager()

    # Load dataset
    x_train, x_test, y_train, y_test = load_data()

    # Train and log models
    model_manager.train_and_log_models(x_train, y_train, x_test, y_test)

    # Test the prediction endpoint
    url = "http://127.0.0.1:9696/predict"
    headers = {"Content-Type": "application/json"}
    sample = x_test.iloc[0].to_dict()

    response = requests.post(url, json=sample, headers=headers)
    response_data = response.json()

    assert response.status_code == 200
    assert 'MEDV' in response_data
    assert 'model_version' in response_data
    assert response_data['model_version'] == RUN_ID


if __name__ == "__main__":
    pytest.main()
