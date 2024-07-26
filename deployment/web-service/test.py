import requests

# Define the test input data
test_row = {
    'LSTAT': 9.04,
    'INDUS': 4.05,
    'NOX': 0.51,
    'PTRATIO': 16.6,
    'RM': 6.416,
    'TAX': 296.0,
    'DIS': 2.6463,
    'AGE': 84.1
}

# Send a POST request to the prediction endpoint
response = requests.post('http://127.0.0.1:9696/predict', json=test_row)

# Send a POST request to the prediction endpoint
# response = requests.post('https://house-price-prediction-service-7ca39ab862fd.herokuapp.com/predict', json=test_row)
result = response.json()

print(result)
