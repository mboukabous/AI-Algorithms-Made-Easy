
"""
CatBoost Classifier setup.

Features:
- Uses `CatBoostClassifier`.
- Handles categorical features natively but we still rely on pipeline encoding.
- Good for both binary and multi-class.
- Default scoring: 'accuracy'.

Requires `catboost` installed.
"""

from catboost import CatBoostClassifier

estimator = CatBoostClassifier(verbose=0, random_state=42)

param_grid = {
    'model__iterations': [100],
    'model__depth': [3, 5],
    'model__learning_rate': [0.01, 0.1],
    # Preprocessing params
    #'preprocessor__num__imputer__strategy': ['mean','median'],
    #'preprocessor__num__scaler__with_mean': [True,False],
    #'preprocessor__num__scaler__with_std': [True,False],
}

default_scoring = 'accuracy'
