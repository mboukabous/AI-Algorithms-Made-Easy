# Utils

This directory contains utility scripts and helper functions that are used throughout the project. These scripts provide common functionalities such as data preprocessing, hyperparameter tuning, and other support functions that assist in model training and evaluation for both regression and classification tasks.

## Contents

- [`supervised_hyperparameter_tuning.py`](#supervised_hyperparameter_tuningpy)

### `supervised_hyperparameter_tuning.py`

This script contains functions for performing hyperparameter tuning on supervised learning models (both regression and classification) using scikit-learn's `Pipeline` and `GridSearchCV`.

#### Functions

- **`regression_hyperparameter_tuning(X, y, estimator, param_grid, cv=5, scoring=None)`**

  Performs hyperparameter tuning for regression models.
  
  **Parameters:**
  - `X`: Feature matrix (pd.DataFrame).
  - `y`: Numeric target variable (pd.Series).
  - `estimator`: A scikit-learn regressor (e.g., `LinearRegression()`).
  - `param_grid`: Dict with parameter names and lists of values.
  - `cv`: Number of cross-validation folds (default 5).
  - `scoring`: Scoring metric (e.g. 'neg_root_mean_squared_error').

  **Returns:**
  - `best_model`: Pipeline with best found hyperparameters.
  - `best_params`: Dictionary of best hyperparameters.

- **`classification_hyperparameter_tuning(X, y, estimator, param_grid, cv=5, scoring=None)`**

  Performs hyperparameter tuning for classification models.
  
  **Parameters:**
  - `X`: Feature matrix (pd.DataFrame).
  - `y`: Target variable for classification (pd.Series), can be binary or multi-class.
  - `estimator`: A scikit-learn classifier (e.g., `LogisticRegression()`, `RandomForestClassifier()`).
  - `param_grid`: Dict with parameter names and lists of values.
  - `cv`: Number of cross-validation folds (default 5).
  - `scoring`: Scoring metric (e.g. 'accuracy', 'f1', 'f1_macro', 'roc_auc').

  **Returns:**
  - `best_model`: Pipeline with best found hyperparameters.
  - `best_params`: Dictionary of best hyperparameters.

#### Usage Examples

**Regression Example:**
```python
from utils.supervised_hyperparameter_tuning import regression_hyperparameter_tuning
from sklearn.linear_model import LinearRegression

X = ...  # Your regression features
y = ...  # Your numeric target variable
param_grid = {
    'model__fit_intercept': [True, False]
    # Add other parameters if needed
}

best_model, best_params = regression_hyperparameter_tuning(X, y, LinearRegression(), param_grid, scoring='neg_root_mean_squared_error')
```

**Classification Example (Binary or Multi-Class):**
```python
from utils.supervised_hyperparameter_tuning import classification_hyperparameter_tuning
from sklearn.ensemble import RandomForestClassifier

X = ...  # Your classification features
y = ...  # Your categorical target variable (binary or multi-class)
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10]
}

best_model, best_params = classification_hyperparameter_tuning(X, y, RandomForestClassifier(), param_grid, scoring='accuracy')
```
