FROM python:3.9.12-slim

# Upgrade pip and install pipenv in a single RUN command
RUN pip install -U pip pipenv

WORKDIR /app

# Copy Pipfile and Pipfile.lock together
COPY Pipfile Pipfile.lock ./

# Install dependencies from Pipfile.lock
RUN pipenv install --dev --deploy

# Copy the remaining files
COPY predict.py lin_reg.bin ./

# Expose port 9696
EXPOSE 9696

# Use gunicorn to serve the application
CMD ["pipenv", "run", "gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]
