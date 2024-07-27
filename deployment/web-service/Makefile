# Makefile

# Variables
VENV = $(CURDIR)/venv
PIPENV = pipenv

# Commands
PIPENV_RUN = $(PIPENV) run
PYTEST = $(PIPENV_RUN) pytest
PYLINT = $(PIPENV_RUN) pylint
BLACK = $(PIPENV_RUN) black
ISORT = $(PIPENV_RUN) isort
PRE_COMMIT = $(PIPENV_RUN) pre-commit

# Paths
UNIT_TESTS = deployment/web-service/test_unittest.py
INTEGRATION_TESTS = deployment/web-service/test_integration.py

# Targets
.PHONY: all setup venv install-deps lint format test unit-test integration-test pre-commit-install

all: setup install-deps lint format test

setup: venv

venv:
	$(PIPENV) --python 3.8
	$(PIPENV) install --dev

install-deps:
	$(PIPENV) install --dev

lint:
	$(PYLINT) $(UNIT_TESTS) $(INTEGRATION_TESTS)

format:
	$(BLACK) .
	$(ISORT) .

test: unit-test integration-test

unit-test:
	$(PYTEST) -v --tb=short $(UNIT_TESTS)

integration-test:
	$(PYTEST) -v --tb=short $(INTEGRATION_TESTS)

pre-commit-install:
	$(PRE_COMMIT) install

# Show differences after formatting/sorting
format-diff:
	$(BLACK) --diff .
	$(ISORT) --diff .

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -r {} +
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .pylint.d
	rm -rf .coverage

# Initialize git repository
init-git:
	git init
	$(PIPENV) install --dev pre-commit
	$(PRE_COMMIT) sample-config > .pre-commit-config.yaml
	$(PRE_COMMIT) install
	git add .
	git commit -m "Initial commit with pre-commit hooks"

