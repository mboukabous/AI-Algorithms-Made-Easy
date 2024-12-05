
"""
This module sets up an Extra Trees Regressor with hyperparameter tuning.

Features:
- Uses `ExtraTreesRegressor` estimator from scikit-learn.
- Defines a hyperparameter grid for ensemble parameters.
- Similar to Random Forest but uses random thresholds for splitting.

Special Considerations:
- Not sensitive to feature scaling.
- Can handle large datasets efficiently.
- Less prone to overfitting compared to single decision trees.
"""

from sklearn.ensemble import ExtraTreesRegressor

# Define the estimator
estimator = ExtraTreesRegressor(random_state=42, n_jobs=-1)

# Define the hyperparameter grid
param_grid = {
    'model__n_estimators': [100, 200],  # Common range for estimators
    'model__criterion': ['squared_error'],  # Focus on the most widely used criterion
    'model__max_depth': [None, 10, 20],  # Unrestricted depth and reasonable constraints
    'model__min_samples_split': [2, 5],  # Commonly used values
    'model__min_samples_leaf': [1, 2],  # Prevent overfitting with larger leaves
    'model__max_features': ['sqrt', 'log2'],  # Reduce to most common feature sampling strategies
    'preprocessor__num__imputer__strategy': ['mean'],  # Single imputation strategy
}

# Optional: Define the default scoring metric
default_scoring = 'neg_root_mean_squared_error'
