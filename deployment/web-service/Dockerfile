# Use an official Python runtime as a parent image
FROM python:3.9.12-slim

# Upgrade pip and install pipenv
RUN pip install -U pip pipenv

# Set the working directory
WORKDIR /app

# Copy Pipfile and Pipfile.lock together
COPY Pipfile Pipfile.lock ./

# Install dependencies from Pipfile.lock
RUN pipenv install --dev --deploy --ignore-pipfile

# Copy the remaining files
COPY predict.py test.py ./

# Expose port 9696
EXPOSE 9696

# Use gunicorn to serve the application
CMD ["pipenv", "run", "gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]
