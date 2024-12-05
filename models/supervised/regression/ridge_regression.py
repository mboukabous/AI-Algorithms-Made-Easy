
"""
This module sets up a Ridge Regression model with hyperparameter tuning.

Features:
- Uses `Ridge` estimator from scikit-learn.
- Defines a hyperparameter grid for preprocessing and model-specific parameters.
- Addresses potential convergence warnings by increasing `max_iter`.
- Considers solvers compatible with dense data after modifying `OneHotEncoder`.

Special Considerations:
- Ridge Regression may produce convergence warnings if `max_iter` is insufficient.
- Applying a log transformation (`log_transform`) to the target variable can be beneficial if it's skewed.
- Ensure `OneHotEncoder` outputs dense arrays to avoid solver compatibility issues.
"""

from sklearn.linear_model import Ridge

# Define the estimator
estimator = Ridge()

# Define the hyperparameter grid
param_grid = {
    'model__alpha': [0.1, 1.0, 10.0],
    'model__solver': ['auto', 'svd', 'cholesky'],
    'model__max_iter': [1000, 5000],
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'preprocessor__num__scaler__with_mean': [True, False],
    'preprocessor__num__scaler__with_std': [True, False],
}

# Optional: Define the default scoring metric
default_scoring = 'neg_root_mean_squared_error'
