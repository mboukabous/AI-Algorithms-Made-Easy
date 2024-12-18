
"""
K-Nearest Neighbors Classifier setup.

Features:
- Uses `KNeighborsClassifier`.
- Works for binary and multi-class tasks.
- Default scoring: 'accuracy'.

Considerations:
- `n_neighbors`, `weights`, and `p` (Minkowski distance) are common parameters to tune.
"""

from sklearn.neighbors import KNeighborsClassifier

estimator = KNeighborsClassifier()

param_grid = {
    'model__n_neighbors': [3, 5],  # Reduced to two neighbor options
    'model__weights': ['uniform'],  # Focused on one weighting strategy
    'model__p': [2],  # Fixed to Euclidean distance
    # Preprocessing params
    #'preprocessor__num__imputer__strategy': ['mean'],
    #'preprocessor__num__scaler__with_mean': [True],
    #'preprocessor__num__scaler__with_std': [True],
}

default_scoring = 'accuracy'
