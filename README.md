Dataset Description
The dataset used in this project is the Boston Housing Dataset from the UCI Machine Learning Repository. The data was collected in 1978 and contains information on 506 housing entries from various suburbs in Boston. Each entry includes 14 features that describe different aspects of the suburbs.

Features Description
Here is a detailed description of each feature in the dataset:

CRIM: Per capita crime rate by town. A higher value indicates a higher crime rate.
ZN: Proportion of residential land zoned for lots larger than 25,000 sq.ft. Indicates the percentage of land for large residential plots.
INDUS: Proportion of non-retail business acres per town. Reflects the level of industrial activity in the town.
CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise). Indicates whether the town is situated near the Charles River.
NOX: Nitric oxides concentration (parts per 10 million). Represents the level of air pollution.
RM: Average number of rooms per dwelling. Reflects the average house size.
AGE: Proportion of owner-occupied units built prior to 1940. Indicates the age of the housing stock.
DIS: Weighted distances to five Boston employment centers. Reflects accessibility to employment opportunities.
RAD: Index of accessibility to radial highways. A higher index indicates better access to major roads.
TAX: Full-value property tax rate per $10,000. Reflects the local tax rate.
PTRATIO: Pupil-teacher ratio by town. Indicates the quality of education services.
B: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town. Reflects racial demographics.
LSTAT: Percentage of lower status of the population. Indicates the socioeconomic status of the population.
MEDV: Median value of owner-occupied homes in $1000s. This is the target variable for the prediction task.

Data Preprocessing
The preprocessing steps include:

Initialization and Experiment Management
Set up and ensure the active status of MLflow experiments.
Restore experiments if they are deleted.

Data Loading and Preprocessing
Load the housing dataset from a URL.
Split the dataset into training and testing sets.
Preprocess data using transformations such as logarithmic scaling and normalization.

Model Training
Create preprocessing and model training pipelines using Pipeline and ColumnTransformer.
Train models and log the training runs with MLflow, including metrics such as RMSE.

Model Evaluation
Find the best run based on specified metrics.
Load models from MLflow and evaluate them on test data.
Compute and print the RMSE for model evaluation.
Model Registration and Promotion

Register trained models in MLflow.
Promote models to specific stages (e.g., Production).

Model Download
Download models from MLflow to a local directory for deployment or further use.

Flask Application
This module contains the Flask application for house price prediction.

Application Details
GitHub URL: The raw model file is fetched from GitHub.
RUN_ID: Versioning identifier for the model.
Model loaded from GITHUB_RAW_URL = (
    "https://raw.githubusercontent.com/arunv22/zoomcamp_mlops_project/main/model.pkl"
)
The Flask application provides a simple API for predicting house prices: deployment/web-service/predict.py
/predict: API endpoint to get house price predictions based on input features.
It expects a POST request with JSON data containing the input features.

response fetched from the model deployed to heroku app endpoint hosted: deployment/web-service/test.py
requests.post('https://house-price-prediction-service-7ca39ab862fd.herokuapp.com/predict', json=test_row)

Best Practices used include:

Docker - mlops-mage/docker-compose.yml ; docker script can be run from mlops-mage/scripts/start.sh

Unit Tests - deployment/web-service/test_unittest.py
The project includes unit tests to ensure that individual components of the system function correctly. These tests validate the behavior of isolated pieces of code, such as data preprocessing functions and model training methods.

Integration Test
An integration test is included to verify the correct interaction between different components of the system. This test checks the end-to-end functionality of the pipeline, from data loading and preprocessing to model training and evaluation.

Linter and Code Formatter
To maintain code quality and consistency, pylint and code formatter are used:

Makefile - deployment/web-service/Makefile
A Makefile is used to automate common tasks, such as running tests and setting up the development environment. This helps streamline the development workflow and ensures that common tasks are performed consistently.

Pre-commit Hooks - deployment/web-service/.pre-commit-config.yaml
Pre-commit hooks are configured to automatically run linting and formatting checks before committing changes to the repository. This helps catch issues early and maintain code quality.

CI/CD Pipeline - 
.github/workflows/ci-tests.yml
.github/workflows/cd-deploy.yml
A Continuous Integration and Continuous Deployment (CI/CD) pipeline is set up using GitHub Actions. This pipeline automates the following processes:
