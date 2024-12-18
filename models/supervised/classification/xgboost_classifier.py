
"""
XGBoost Classifier setup.

Features:
- Uses `XGBClassifier` from xgboost library.
- Excellent performance for binary and multi-class tasks.
- Default scoring: 'accuracy'.

Note: Ensure `xgboost` is installed.
"""

from xgboost import XGBClassifier

estimator = XGBClassifier(eval_metric='logloss', random_state=42)

param_grid = {
    'model__n_estimators': [100],
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.01, 0.1],
    # Preprocessing params
    #'preprocessor__num__imputer__strategy': ['mean','median'],
    #'preprocessor__num__scaler__with_mean': [True,False],
    #'preprocessor__num__scaler__with_std': [True,False],
}

default_scoring = 'accuracy'
