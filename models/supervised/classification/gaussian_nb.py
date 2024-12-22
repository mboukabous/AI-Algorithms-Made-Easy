
"""
Gaussian Naive Bayes Classifier setup.

Features:
- Uses `GaussianNB`.
- Suitable for binary and multi-class.
- Default scoring: 'accuracy'.

Considerations:
- `var_smoothing` is often the only parameter to tune.
"""

from sklearn.naive_bayes import GaussianNB

estimator = GaussianNB()

param_grid = {
    'model__var_smoothing': [1e-1, 1e-3, 1e-5, 1e-7, 1e-9],
    # Preprocessing params
    #'preprocessor__num__imputer__strategy': ['mean','median'],
    #'preprocessor__num__scaler__with_mean': [True,False],
    #'preprocessor__num__scaler__with_std': [True,False],
}

default_scoring = 'accuracy'
