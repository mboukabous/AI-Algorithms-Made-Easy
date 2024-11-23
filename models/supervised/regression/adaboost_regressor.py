
"""
This module sets up an AdaBoost Regressor with hyperparameter tuning.

Features:
- Uses `AdaBoostRegressor` estimator from scikit-learn.
- Defines a hyperparameter grid for boosting parameters.
- Combines weak learners to form a strong predictor.

Special Considerations:
- Sensitive to outliers.
- Not sensitive to feature scaling.
- Base estimator is a Decision Tree by default.
"""

from sklearn.ensemble import AdaBoostRegressor

# Define the estimator
estimator = AdaBoostRegressor(random_state=42)

# Define the hyperparameter grid
param_grid = {
    'model__n_estimators': [50, 100],  # Focus on a narrower range of estimators
    'model__learning_rate': [0.001, 0.01, 0.1, 1.0],  # Keep a good spread for learning rates
    'model__loss': ['linear'],  # Focus on the most commonly used loss function
    'preprocessor__num__imputer__strategy': ['mean'],  # Single imputation strategy
}

# Optional: Define the default scoring metric
default_scoring = 'neg_root_mean_squared_error'
