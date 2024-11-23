
"""
This module defines the setup for performing Linear Regression with hyperparameter tuning.

Features:
- Sets up a `LinearRegression` estimator from scikit-learn.
- Defines a hyperparameter grid for preprocessing and model-specific parameters.
- Specifies an optional default scoring metric for evaluating the model.

Special Considerations:
- Linear Regression doesn't typically require special handling.
- Applying a log transformation to the target variable (`log_transform`) can be beneficial if it's skewed.
"""

from sklearn.linear_model import LinearRegression

# Define the estimator
estimator = LinearRegression()

# Define the hyperparameter grid
param_grid = {
    'model__fit_intercept': [True, False],
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'preprocessor__num__scaler__with_mean': [True, False],
    'preprocessor__num__scaler__with_std': [True, False],
}

# Optional: Define the default scoring metric
default_scoring = 'neg_root_mean_squared_error'
