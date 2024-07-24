import predict

# test_row = {
#     'LSTAT': 9.04,
#     'INDUS': 4.05,
#     'NOX': 0.51,
#     'PTRATIO': 16.6,
#     'RM': 6.416,
#     'TAX': 296.0,
#     'DIS': 2.6463,
#     'AGE': 84.1
# }

# test_row = {'LSTAT': 8.94,
#  'INDUS': 12.83,
#  'NOX': 0.437,
#  'PTRATIO': 18.7,
#  'RM': 6.286,
#  'TAX': 398.0,
#  'DIS': 4.5026,
#  'AGE': 45.0}
#
# pred = predict.predict(test_row)
# print(pred)


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
result = response.json()

print(result)
