
"""
This module defines the setup for performing linear regression with hyperparameter tuning.

Features:
- Sets up a `LinearRegression` estimator from scikit-learn.
- Defines a hyperparameter grid for preprocessing and model-specific parameters.
- Specifies an optional default scoring metric for evaluating the model.

Usage:
This module is intended to be used with a pipeline that includes preprocessing steps
(e.g., imputing missing values, scaling numeric data, and encoding categorical variables)
and a hyperparameter tuning method such as GridSearchCV.
"""

from sklearn.linear_model import LinearRegression

# Define the estimator
estimator = LinearRegression()

# Define the hyperparameter grid
param_grid = {
    'model__fit_intercept': [True, False],
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'preprocessor__num__scaler__with_mean': [True, False],
    'preprocessor__num__scaler__with_std': [True, False]
}

# Optional: Define the default scoring metric
default_scoring = 'neg_root_mean_squared_error'
