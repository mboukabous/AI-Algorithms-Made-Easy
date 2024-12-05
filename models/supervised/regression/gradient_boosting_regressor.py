
"""
This module sets up a Gradient Boosting Regressor with hyperparameter tuning.

Features:
- Uses `GradientBoostingRegressor` estimator from scikit-learn.
- Defines a hyperparameter grid for boosting parameters.
- Builds sequential models to minimize errors.

Special Considerations:
- Sensitive to overfitting; tune `n_estimators` and `learning_rate`.
- Not sensitive to feature scaling.
- Longer training times compared to other models.
"""

from sklearn.ensemble import GradientBoostingRegressor

# Define the estimator
estimator = GradientBoostingRegressor(random_state=42)

# Define the hyperparameter grid
param_grid = {
    'model__n_estimators': [100, 200],  # Focused range of estimators
    'model__learning_rate': [0.001, 0.01, 0.1, 1],  # Commonly used learning rates
    'model__max_depth': [3, 5],  # Standard depth values
    'model__subsample': [0.8],  # Single value to focus on speed
    'model__min_samples_split': [2],  # Default value
    'model__min_samples_leaf': [1],  # Default value
    'preprocessor__num__imputer__strategy': ['mean'],  # Single imputation strategy
}

# Optional: Define the default scoring metric
default_scoring = 'neg_root_mean_squared_error'
