
"""
This module sets up a Support Vector Regressor (SVR) with hyperparameter tuning.

Features:
- Uses `SVR` estimator from scikit-learn.
- Defines a hyperparameter grid for kernel parameters.
- Effective in high-dimensional spaces.

Special Considerations:
- Feature scaling is crucial for SVR.
- Training time can be significant for large datasets.
- Applying a log transformation (`log_transform`) can be beneficial if the target variable is skewed.
"""

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Define the estimator
estimator = SVR()

# Define the hyperparameter grid
param_grid = {
    'model__kernel': ['rbf'],  # Stick to the most effective kernel
    'model__C': [0.1, 1.0, 10.0],  # Focus on a narrower range
    'model__epsilon': [0.1, 0.2, 0.5],  # Retain small deviations
    'model__gamma': ['scale', 0.1],  # Simplify gamma
    'preprocessor__num__imputer__strategy': ['mean'],  # Single imputation strategy
    'preprocessor__num__scaler__with_mean': [True],
    'preprocessor__num__scaler__with_std': [True],
}

# Optional: Define the default scoring metric
default_scoring = 'neg_root_mean_squared_error'
