import os
import requests
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from io import BytesIO

# Define the GitHub URL for the raw model file
GITHUB_RAW_URL = "https://raw.githubusercontent.com/arunv22/zoomcamp_mlops_project/main/model.pkl"

# Define the run ID for versioning (you can update this if needed)
RUN_ID = 'ec53532a97e74b2eb5c4a1eeb6834be2'

# Fetch the model file from GitHub and load it into memory
response = requests.get(GITHUB_RAW_URL)
if response.status_code == 200:
    model_file = BytesIO(response.content)
    model = pickle.load(model_file)
    print("Model loaded successfully from GitHub")
else:
    raise Exception(f"Failed to fetch model: {response.status_code} - {response.text}")


# Define the predict function
def predict(features):
    """
    Predict house prices based on the input features.

    Args:
        features (dict): A dictionary containing the input features for the model.

    Returns:
        float: The predicted house price.
    """
    df = pd.DataFrame([features])
    preds = model.predict(df)
    preds = np.expm1(preds)  # Transform log predictions back to original scale
    return preds[0]


# Initialize the Flask app
app = Flask('house-price-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    API endpoint to get house price predictions.

    Request should be a JSON object containing the input features.

    Returns:
        JSON: A JSON response containing the predicted house price and model version.
    """
    features = request.get_json()
    pred = predict(features)

    result = {
        'MEDV': pred,
        'model_version': RUN_ID  # You can update this if needed
    }

    return jsonify(result)


@app.route('/')
def home():
    """
    Home route to provide information about the service.

    Returns:
        str: A welcome message.
    """
    return "Welcome to the house price prediction service. Use POST /predict to get predictions."


if __name__ == "__main__":
    from waitress import serve

    port = int(os.environ.get('PORT', 9696))  # Get the port from the environment variable or use 9696 as default
    serve(app, host='0.0.0.0', port=port)  # Start the server using waitress
