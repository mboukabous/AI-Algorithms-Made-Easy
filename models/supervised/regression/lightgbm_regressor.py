
"""
This module sets up a LightGBM Regressor with hyperparameter tuning.

Features:
- Uses `LGBMRegressor` estimator from LightGBM.
- Defines a hyperparameter grid for boosting parameters.
- Optimized for speed and performance.

Special Considerations:
- Requires the `lightgbm` library (`pip install lightgbm`).
- Can handle categorical features if provided appropriately.
- Not sensitive to feature scaling.
"""

from lightgbm import LGBMRegressor

# Define the estimator
estimator = LGBMRegressor(
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# Define hyperparameter grid
param_grid = {
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.01, 0.05],
    'model__num_leaves': [15, 31],
    'model__max_depth': [10, 20],
    'model__min_data_in_leaf': [20, 50],
    'model__colsample_bytree': [0.8],
    'preprocessor__num__imputer__strategy': ['mean'],
}

# Optional: Define the default scoring metric
default_scoring = 'neg_root_mean_squared_error'
