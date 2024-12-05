
"""
This module sets up an XGBoost Regressor with hyperparameter tuning.

Features:
- Uses `XGBRegressor` estimator from XGBoost.
- Defines a hyperparameter grid for boosting parameters.
- Efficient and scalable implementation of gradient boosting.

Special Considerations:
- Requires the `xgboost` library (`pip install xgboost`).
- Handles missing values internally.
- Not sensitive to feature scaling.
- May require setting `tree_method` to 'gpu_hist' for GPU acceleration if available.
"""

from xgboost import XGBRegressor

# Define the estimator
estimator = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)

# Define the hyperparameter grid
param_grid = {
    'model__n_estimators': [100, 200],  # Common range for estimators
    'model__learning_rate': [0.05, 0.1],  # Common learning rates
    'model__max_depth': [3, 5],  # Typical depth for gradient boosting
    'model__subsample': [0.8],  # Fixed subsample value to reduce complexity
    'model__colsample_bytree': [0.8],  # Fixed colsample value to reduce complexity
    'model__reg_alpha': [0, 0.1],  # Focus on smaller values for L1 regularization
    'model__reg_lambda': [1],  # Default L2 regularization
    'preprocessor__num__imputer__strategy': ['mean'],  # Single imputation strategy
}

# Optional: Define the default scoring metric
default_scoring = 'neg_root_mean_squared_error'
