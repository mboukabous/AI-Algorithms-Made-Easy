
"""
This module sets up a CatBoost Regressor with hyperparameter tuning.

Features:
- Uses `CatBoostRegressor` estimator from CatBoost.
- Defines a hyperparameter grid for boosting parameters.
- Handles categorical features natively.

Special Considerations:
- Requires the `catboost` library (`pip install catboost`).
- Adjust the preprocessing pipeline to skip encoding categorical features.
- Not sensitive to feature scaling.
- Can be slower to train compared to other boosting algorithms.
"""

from catboost import CatBoostRegressor

# Define the estimator
estimator = CatBoostRegressor(random_state=42, verbose=0)

# Define the hyperparameter grid
param_grid = {
    'model__iterations': [500],  # Fixed to a reasonable value for faster tuning
    'model__learning_rate': [0.05, 0.1],  # Common learning rates
    'model__depth': [6, 8],  # Typical depths for balance between speed and accuracy
    'model__l2_leaf_reg': [3],  # Most impactful regularization value
    'preprocessor__num__imputer__strategy': ['mean'],  # Single imputation strategy
}

# Optional: Define the default scoring metric
default_scoring = 'neg_root_mean_squared_error'
