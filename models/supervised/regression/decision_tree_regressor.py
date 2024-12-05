
"""
This module sets up a Decision Tree Regressor with hyperparameter tuning.

Features:
- Uses `DecisionTreeRegressor` estimator from scikit-learn.
- Defines a hyperparameter grid for tree-specific parameters.
- Handles non-linear relationships and interactions.

Special Considerations:
- Decision Trees are not affected by feature scaling.
- Can easily overfit; control tree depth and splitting criteria.
- No need for scaling transformers in the preprocessing pipeline.
"""

from sklearn.tree import DecisionTreeRegressor

# Define the estimator
estimator = DecisionTreeRegressor(random_state=42)

# Define the hyperparameter grid
param_grid = {
    'model__criterion': ['squared_error', 'absolute_error'],  # Only two key criteria
    'model__max_depth': [5, 10, 20, None],  # Depth variations
    'model__min_samples_split': [2, 10],  # Commonly used values
    'model__min_samples_leaf': [1, 4],  # Few values for leaves
    'preprocessor__num__imputer__strategy': ['mean'],  # Focused on a single strategy
}

# Optional: Define the default scoring metric
default_scoring = 'neg_root_mean_squared_error'
