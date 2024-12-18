
"""
Linear Discriminant Analysis (LDA) Classifier setup.

Features:
- Uses `LinearDiscriminantAnalysis`.
- Works for binary and multi-class tasks.
- Default scoring: 'accuracy'.

Considerations:
- `solver` can be tuned.
- Some solvers allow `shrinkage` parameter.
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

estimator = LinearDiscriminantAnalysis()

param_grid = {
    'model__solver': ['svd', 'lsqr'],
    # If solver='lsqr', can tune shrinkage parameter if needed
    # Preprocessing params
    #'preprocessor__num__imputer__strategy': ['mean','median'],
    #'preprocessor__num__scaler__with_mean': [True,False],
    #'preprocessor__num__scaler__with_std': [True,False],
}

default_scoring = 'accuracy'
