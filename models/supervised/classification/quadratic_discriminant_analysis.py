
"""
Quadratic Discriminant Analysis (QDA) Classifier setup.

Features:
- Uses `QuadraticDiscriminantAnalysis`.
- Works for binary and multi-class tasks.
- Default scoring: 'accuracy'.

Considerations:
- `reg_param` can be tuned to control regularization.
"""

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

estimator = QuadraticDiscriminantAnalysis()

param_grid = {
    'model__reg_param': [0.0, 0.1, 0.5],
    # Preprocessing params
    #'preprocessor__num__imputer__strategy': ['mean','median'],
    #'preprocessor__num__scaler__with_mean': [True,False],
    #'preprocessor__num__scaler__with_std': [True,False],
}

default_scoring = 'accuracy'
