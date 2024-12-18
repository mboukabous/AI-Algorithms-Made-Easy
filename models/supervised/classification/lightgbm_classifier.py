
"""
LightGBM Classifier setup.

Features:
- Uses `LGBMClassifier`.
- Fast and efficient for binary and multi-class tasks.
- Default scoring: 'accuracy'.

Requires `lightgbm` installed.
"""

from lightgbm import LGBMClassifier

estimator = LGBMClassifier(verbose=-1, random_state=42)

param_grid = {
    'model__n_estimators': [100],
    'model__num_leaves': [31, 63],
    'model__learning_rate': [0.01, 0.1],
    # Preprocessing params
    #'preprocessor__num__imputer__strategy': ['mean','median'],
    #'preprocessor__num__scaler__with_mean': [True,False],
    #'preprocessor__num__scaler__with_std': [True,False],
}

default_scoring = 'accuracy'
