
"""
This module sets up a K-Nearest Neighbors Regressor with hyperparameter tuning.

Features:
- Uses `KNeighborsRegressor` estimator from scikit-learn.
- Defines a hyperparameter grid for neighbor parameters.
- Non-parametric method useful for capturing local patterns.

Special Considerations:
- Feature scaling is crucial for KNN.
- Sensitive to the choice of `n_neighbors`.
- Training is fast, but prediction can be slow on large datasets.
"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Define the estimator
estimator = KNeighborsRegressor(n_jobs=-1)

# Define the hyperparameter grid
param_grid = {
    'model__n_neighbors': [3, 5, 7],  # Focus on common neighbor values
    'model__weights': ['uniform', 'distance'],  # Standard options
    'model__algorithm': ['auto', 'ball_tree'],  # Reduce algorithms to commonly used ones
    'model__p': [1, 2],  # Manhattan and Euclidean distances
    'preprocessor__num__imputer__strategy': ['mean'],  # Single imputation strategy
    'preprocessor__num__scaler__with_mean': [True],  # StandardScaler
    'preprocessor__num__scaler__with_std': [True],  # StandardScaler
}

# Optional: Define the default scoring metric
default_scoring = 'neg_root_mean_squared_error'
