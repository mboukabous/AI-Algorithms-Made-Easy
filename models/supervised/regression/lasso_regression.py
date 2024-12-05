
"""
This module sets up a Lasso Regression model with hyperparameter tuning.

Features:
- Uses `Lasso` estimator from scikit-learn.
- Defines a hyperparameter grid for preprocessing and model-specific parameters.
- Increases `max_iter` to address convergence warnings.

Special Considerations:
- Lasso Regression may produce convergence warnings if `max_iter` is insufficient.
- Applying a log transformation (`log_transform`) to the target variable can be beneficial if it's skewed.
- Ensure `OneHotEncoder` outputs dense arrays to avoid compatibility issues.
"""

from sklearn.linear_model import Lasso

# Define the estimator
estimator = Lasso()

# Define the hyperparameter grid
param_grid = {
    'model__alpha': [0.01, 0.1, 1.0, 10.0],  # Regularization strength
    'model__max_iter': [5000],  # Single value to ensure convergence
    'model__fit_intercept': [True],  # Assume the intercept is important
    'model__selection': ['cyclic'],  # Focus on the default cyclic selection
    'preprocessor__num__imputer__strategy': ['mean'],  # Single imputation strategy
    'preprocessor__num__scaler__with_mean': [True],  # StandardScaler
    'preprocessor__num__scaler__with_std': [True],  # StandardScaler
}

# Optional: Define the default scoring metric
default_scoring = 'neg_root_mean_squared_error'
