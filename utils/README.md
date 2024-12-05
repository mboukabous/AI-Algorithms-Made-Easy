# Utils

This directory contains utility scripts and helper functions that are used throughout the project. These scripts provide common functionalities such as data preprocessing, hyperparameter tuning, and other support functions that assist in model training and evaluation.

## Contents

- [`supervised_hyperparameter_tuning.py`](#supervised_hyperparameter_tuning.py)

### `supervised_hyperparameter_tuning.py`

This script contains functions for performing hyperparameter tuning on supervised learning models using scikit-learn's `Pipeline` and `GridSearchCV`.

#### Functions

- **`regression_hyperparameter_tuning(X_train, y_train, estimator, param_grid, cv=5, scoring=None)`**

  Performs hyperparameter tuning using grid search cross-validation.

  - **Parameters:**
    - `X_train`: Training features.
    - `y_train`: Training target variable.
    - `estimator`: A scikit-learn estimator (e.g., `LinearRegression()`).
    - `param_grid`: Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values.
    - `cv`: Number of cross-validation folds. Default is 5.
    - `scoring`: Scoring metric to use. Default depends on the estimator.

  - **Returns:**
    - `best_model`: The estimator with the best found parameters.
    - `best_params`: Dictionary of the best parameters.

#### Usage Example

```python
from utils.supervised_hyperparameter_tuning import regression_hyperparameter_tuning
from sklearn.linear_model import LinearRegression

# Define estimator and parameter grid
estimator = LinearRegression()
param_grid = {
    'model__fit_intercept': [True, False],
    # Add other parameters
}

# Perform hyperparameter tuning
best_model, best_params = regression_hyperparameter_tuning(X_train, y_train, estimator, param_grid)
