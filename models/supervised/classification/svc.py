
"""
Support Vector Classifier setup.

Features:
- Uses `SVC` from scikit-learn.
- Handles binary classification naturally, and multi-class via OvR by default.
- Default scoring: 'accuracy'.

Considerations:
- `C` and `kernel` are key parameters.
- If `kernel='rbf'`, also tune `gamma`.
"""

from sklearn.svm import SVC

estimator = SVC(random_state=42)

param_grid = {
    'model__C': [0.1, 1.0],  # Reduced the range
    'model__kernel': ['linear'],  # Focused on linear kernel
    'model__gamma': ['scale'],  # Fixed the gamma to one option
    # Preprocessing params
    #'preprocessor__num__imputer__strategy': ['mean'],
    #'preprocessor__num__scaler__with_mean': [True],
    #'preprocessor__num__scaler__with_std': [True],
}

default_scoring = 'accuracy'
