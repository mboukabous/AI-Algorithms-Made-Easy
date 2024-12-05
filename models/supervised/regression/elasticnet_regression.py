
"""
This module sets up an ElasticNet Regression model with hyperparameter tuning.

Features:
- Uses `ElasticNet` estimator from scikit-learn.
- Combines L1 and L2 regularization.
- Increases `max_iter` to address convergence warnings.

Special Considerations:
- May produce convergence warnings if `max_iter` is insufficient.
- Adjust `l1_ratio` to balance between Lasso and Ridge penalties.
- Applying a log transformation (`log_transform`) to the target variable can be beneficial if it's skewed.
- Ensure `OneHotEncoder` outputs dense arrays.
"""

from sklearn.linear_model import ElasticNet

# Define the estimator
estimator = ElasticNet()

# Define the hyperparameter grid
param_grid = {
    'model__alpha': [0.01, 0.1, 1.0, 10.0],  # Regularization strength
    'model__l1_ratio': [0.2, 0.5, 0.8],  # Balance between L1 (Lasso) and L2 (Ridge)
    'model__max_iter': [5000],  # Sufficient to avoid convergence warnings
    'model__fit_intercept': [True],  # Assume intercept is important
    'model__selection': ['cyclic'],  # Focus on the default cyclic selection
    'preprocessor__num__imputer__strategy': ['mean'],  # Single imputation strategy
    'preprocessor__num__scaler__with_mean': [True],  # StandardScaler
    'preprocessor__num__scaler__with_std': [True],  # StandardScaler
}

# Optional: Define the default scoring metric
default_scoring = 'neg_root_mean_squared_error'
