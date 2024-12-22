
"""
Gradient Boosting Classifier setup.

Features:
- Uses `GradientBoostingClassifier`.
- Great for binary and multi-class tasks.
- Default scoring: 'accuracy'.
"""

from sklearn.ensemble import GradientBoostingClassifier

estimator = GradientBoostingClassifier(random_state=42)

param_grid = {
    'model__n_estimators': [100],
    'model__learning_rate': [0.01, 0.1],
    'model__max_depth': [3],
    # Preprocessing params
    #'preprocessor__num__imputer__strategy': ['mean','median'],
    #'preprocessor__num__scaler__with_mean': [True,False],
    #'preprocessor__num__scaler__with_std': [True,False],
}

default_scoring = 'accuracy'
