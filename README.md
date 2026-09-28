# Linear Regression Architecture Workshop

## Project Overview

This project demonstrates a complete univariate linear regression workflow using housing data.

The workshop starts with loading and exploring data from different sources and then builds a linear regression model using one predictor. Two implementations of linear regression are compared:

1. Linear Regression from scratch using gradient descent.
2. Scikit-Learn Linear Regression.

The project also uses a modular structure, YAML configuration, and experiment tracking to demonstrate basic MLOps practices.

---

## Project Objectives

The main objectives of this project are to:

- Load data from CSV, API, and database sources.
- Explore and understand the datasets using Exploratory Data Analysis (EDA).
- Select a predictor and target variable for linear regression.
- Prepare and split the data for model training and testing.
- Implement linear regression from scratch using gradient descent.
- Train a linear regression model using Scikit-Learn.
- Evaluate and compare both models.
- Organize the machine learning workflow into reusable Python modules.
- Store experiment settings in a YAML configuration file.
- Track model results in a CSV file.
- Create a reproducible machine learning workflow.

---

## Project Structure

```text
Linear-Regression-Architecture-Workshop/
│
├── configs/
│   └── experiment_config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── database/
│
├── experiments/
│   └── results.csv
│
├── notebooks/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluation.py
│   ├── experiment_tracker.py
│   └── run_experiment.py
│
├── linear_regression.ipynb
├── requirements.txt
└── README.md
```

---

## Data Sources

The workshop demonstrates loading data from multiple sources.

### CSV Data

The California Housing dataset is used for the linear regression experiment.

The dataset is stored locally as:

```text
data/raw/california_housing.csv
```

The dataset contains housing-related features that can be used to study the relationship between different variables and median house value.

### API Data

The project also demonstrates retrieving data from an API as part of the data loading and exploration exercise.

The API data is explored separately to understand how external data sources can be accessed and converted into a Pandas DataFrame.

### Database Data

A local SQLite database is also used to demonstrate loading data using SQL.

This shows how the same machine learning project can work with different types of data sources instead of depending only on CSV files.

---

## Exploratory Data Analysis

Exploratory Data Analysis is performed before model training to better understand the dataset.

The analysis includes:

- Viewing dataset dimensions.
- Examining column names and data types.
- Checking for missing values.
- Reviewing descriptive statistics.
- Exploring relationships between variables.
- Creating visualizations to understand the data.

EDA helps identify useful variables before building the regression model.

---

## Linear Regression Experiment

For the main regression experiment, one predictor is used to predict the target variable.

### Predictor

```text
MedInc
```

`MedInc` represents median income.

### Target

```text
MedHouseVal
```

`MedHouseVal` represents median house value.

The purpose of the experiment is to study how median income is related to median house value using a simple univariate linear regression model.

---

## Data Preprocessing

Before training the models, the data is prepared using the preprocessing module.

The preprocessing steps include:

1. Selecting the predictor and target.
2. Removing missing values when necessary.
3. Splitting the dataset into training and testing sets.
4. Standardizing the predictor using `StandardScaler`.

The experiment uses:

```text
Training data: 80%
Testing data: 20%
Random state: 42
```

The scaler is fitted using the training data and then applied to the testing data.

---

## Linear Regression From Scratch

The first model implements linear regression from scratch using gradient descent.

The model starts with initial values for the intercept and slope and updates them repeatedly to reduce prediction error.

The experiment configuration uses:

```text
Learning rate: 0.01
Iterations: 1000
```

This implementation helps demonstrate how linear regression training works internally rather than relying only on a machine learning library.

---

## Scikit-Learn Linear Regression

The second model uses Scikit-Learn's `LinearRegression`.

The same training and testing data are used so that the Scikit-Learn implementation can be compared fairly with the from-scratch implementation.

---

## Model Evaluation

Both models are evaluated using the same regression metrics:

### Mean Squared Error (MSE)

Measures the average squared difference between actual and predicted values.

### Root Mean Squared Error (RMSE)

Represents prediction error in the same general scale as the target variable.

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted values.

### R² Score

Measures how much of the variation in the target variable is explained by the model.

---

## Experiment Results

The final experiment produced the following results:

| Model | MSE | RMSE | MAE | R² |
|---|---:|---:|---:|---:|
| From Scratch Linear Regression | 0.7091 | 0.8421 | 0.6299 | 0.4589 |
| Scikit-Learn Linear Regression | 0.7091 | 0.8421 | 0.6299 | 0.4589 |

The two implementations produced almost identical results.

This shows that the gradient descent implementation from scratch reached approximately the same linear regression solution as Scikit-Learn for this experiment.

The complete experiment results are stored in:

```text
experiments/results.csv
```

---

## Python Modules

The project separates the machine learning workflow into different Python modules.

### `data_loader.py`

Responsible for loading:

- CSV data.
- Database data.
- YAML configuration.

### `preprocessing.py`

Responsible for:

- Selecting the feature and target.
- Removing missing values.
- Splitting training and testing data.
- Standardizing the feature.

### `model.py`

Contains:

- Linear regression from scratch using gradient descent.
- Scikit-Learn Linear Regression.

### `evaluation.py`

Calculates the following regression metrics:

- MSE
- RMSE
- MAE
- R²

### `experiment_tracker.py`

Stores experiment information and model evaluation results in:

```text
experiments/results.csv
```

### `run_experiment.py`

Connects the different modules and runs the complete machine learning pipeline.

The pipeline follows:

```text
Configuration
     ↓
Data Loading
     ↓
Preprocessing
     ↓
Model Training
     ↓
Prediction
     ↓
Evaluation
     ↓
Experiment Tracking
```

---

## Configuration Management

Experiment settings are stored in:

```text
configs/experiment_config.yaml
```

The configuration contains settings such as:

- Dataset path.
- Database path.
- Selected feature.
- Target variable.
- Learning rate.
- Number of iterations.
- Test size.
- Random state.
- Experiment results path.

Using a configuration file reduces the need to change values directly inside the experiment pipeline.

---

## MLOps Architecture

The project follows a simple MLOps-style architecture.

### Separation of Concerns

Different responsibilities are separated into independent Python modules.

For example, data loading, preprocessing, model training, evaluation, and experiment tracking are handled separately.

This makes the code easier to understand, maintain, test, and reuse.

### Config-Driven Experiments

Experiment settings are stored in the YAML configuration instead of being directly written into the experiment runner.

This makes it easier to change experiment settings without changing the main pipeline code.

### Experiment Tracking

Each model run can be stored in:

```text
experiments/results.csv
```

The tracked information includes the model, feature, training settings, and evaluation metrics.

### Reproducibility

The project uses the same dataset, configuration settings, and fixed random state.

This allows the experiment to be rerun using the same settings and helps produce consistent results.

### Real-World Pipeline Alignment

The project separates the machine learning process into stages similar to a larger machine learning system.

The modular structure also makes it easier to replace or extend individual components in future experiments.

---

## Setup

### 1. Create a Virtual Environment

From the project directory, create a Python virtual environment:

```bash
python -m venv .venv
```

### 2. Activate the Virtual Environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Required Packages

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## How to Run the Project

The project should be run from the root project directory.

### Run the Notebook

Open:

```text
linear_regression.ipynb
```

Run the notebook cells in order to view the data exploration and linear regression development.

### Run the Modular Experiment Pipeline

From the project root directory, run:

```bash
python .\src\run_experiment.py
```

The script will:

1. Read the experiment configuration.
2. Load the housing dataset.
3. Preprocess the data.
4. Train the from-scratch linear regression model.
5. Evaluate the from-scratch model.
6. Train the Scikit-Learn model.
7. Evaluate the Scikit-Learn model.
8. Save the experiment results.

The results will be added to:

```text
experiments/results.csv
```

---

## Reproducing the Experiment

To reproduce the experiment:

1. Install the packages from `requirements.txt`.
2. Keep the dataset in the configured data location.
3. Check the settings in `configs/experiment_config.yaml`.
4. Run `src/run_experiment.py` from the project root.
5. Review the generated results in `experiments/results.csv`.

Using the same dataset and configuration should produce consistent results.



## Future Application to Robot Predictive Maintenance

The linear regression approach used in this workshop can later be applied to the robot streaming data in Practical Lab 1.

For predictive maintenance, historical robot sensor data can be used to learn the expected behaviour of the machine. Linear regression can predict an expected sensor value, such as electrical current, based on the available robot data.

When new robot data is streamed, the actual current value can be compared with the value predicted by the regression model.

The planned workflow is:

1. Collect historical robot sensor data.
2. Clean and prepare the sensor data.
3. Select the required features and target value.
4. Train a linear regression model using historical data.
5. Use the model to predict the expected current value.
6. Compare the actual current with the predicted current.
7. Calculate the difference between the actual and predicted values.
8. Use defined thresholds to generate alerts or errors when the difference is unusually large.

This approach can help identify unusual changes in robot current that may indicate a possible machine problem before a failure occurs.

The modular architecture developed in this workshop can also be reused for Practical Lab 1 by adapting the data loading, preprocessing, model training, evaluation, and configuration components for streaming robot data.---

## Conclusion

This workshop demonstrates the complete process of building a simple linear regression experiment.

The project begins with data loading and exploratory data analysis and continues through preprocessing, model training, prediction, and evaluation.

Implementing linear regression both from scratch and with Scikit-Learn helps compare the internal gradient descent approach with a standard machine learning library.

Finally, modular Python files, YAML configuration, experiment tracking, and reproducibility introduce basic MLOps practices that can be extended to larger machine learning projects.