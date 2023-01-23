#!/bin/env bash

# Script run by GitHub Action formatting.yml.
# Dependencies are in `requirements-dev.txt`

python -m black --check --diff .
python -m isort --check --diff .
python -m flake8 .
python -m mypy --python-version 3.7 .
