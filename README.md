# Linear Regression Architecture Workshop

## Project Overview

This project demonstrates univariate linear regression using housing data.

The project includes data loading, preprocessing, model training, model evaluation, configuration management, and experiment tracking.

Two linear regression implementations are used:

1. Linear Regression from scratch using gradient descent.
2. Scikit-Learn Linear Regression.

## Project Structure

- `data/` - Contains raw and processed data.
- `src/` - Contains Python modules.
- `configs/` - Contains experiment configuration.
- `experiments/` - Contains experiment results.
- `LinearRegression.ipynb` - Contains the workshop implementation.

## Python Modules

### data_loader.py

Loads data and project configuration.

### preprocessing.py

Prepares the dataset for machine learning by selecting variables, splitting the data, and standardizing the feature.

### model.py

Contains the custom gradient descent model and Scikit-Learn Linear Regression model.

### evaluation.py

Calculates MSE, RMSE, MAE, and R².

### experiment_tracker.py

Stores experiment settings and model results in a CSV file.

## Configuration

Model settings are stored in:

`configs/experiment_config.yaml`

This includes the selected feature, learning rate, number of iterations, and train/test split.

## Experiment Tracking

Model results are stored in:

`experiments/results.csv`

This makes it possible to compare different experiments.

## Setup

Create and activate a Python virtual environment.

Install the required packages using:

`pip install -r requirements.txt`

Then open and run `LinearRegression.ipynb`.
