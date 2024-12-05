
"""
This module sets up a Random Forest Regressor with hyperparameter tuning.

Features:
- Uses `RandomForestRegressor` estimator from scikit-learn.
- Defines a hyperparameter grid for ensemble parameters.
- Handles non-linear relationships and reduces overfitting through averaging.

Special Considerations:
- Random Forests are robust to outliers and can handle non-linear data.
- Not sensitive to feature scaling.
- Set `n_jobs=-1` to utilize all available CPU cores.
"""

from sklearn.ensemble import RandomForestRegressor

# Define the estimator
estimator = RandomForestRegressor(random_state=42, n_jobs=-1)

# Define the hyperparameter grid
param_grid = {
    'model__n_estimators': [100, 200],  # Focus on a small range of estimators
    'model__max_depth': [10, 20, None],  # Commonly used depth variations
    'model__min_samples_split': [2, 5],  # Commonly used split values
    'model__min_samples_leaf': [1, 2],  # Focused leaf size
    'model__max_features': ['sqrt'],  # "sqrt" is often optimal for Random Forests
    'preprocessor__num__imputer__strategy': ['mean'],  # Single imputation strategy
}

# Optional: Define the default scoring metric
default_scoring = 'neg_root_mean_squared_error'
