import requests
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from io import BytesIO

# Define the GitHub URL for the raw model file
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

# Define the predict function
def predict(features):
    df = pd.DataFrame([features])
    preds = model.predict(df)
    preds = np.expm1(preds)
    return preds[0]

# Initialize the Flask app
app = Flask('house-price-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    features = request.get_json()
    pred = predict(features)

    result = {
        'MEDV': pred,
        'model_version': RUN_ID  # You can update this if needed
    }

    return jsonify(result)

if __name__ == "__main__":
    from waitress import serve
    serve(app, host='0.0.0.0', port=9696)
