
"""
AdaBoost Classifier setup.

Features:
- Uses `AdaBoostClassifier` wrapping a weak learner (by default DecisionTreeClassifier).
- Suitable for binary and multi-class tasks (OvR approach).
- Default scoring: 'accuracy'.
"""

from sklearn.ensemble import AdaBoostClassifier

estimator = AdaBoostClassifier(random_state=42)

param_grid = {
    'model__n_estimators': [100],
    'model__learning_rate': [0.5, 1.0],
    'model__algorithm': ['SAMME'],
    # Preprocessing params
    #'preprocessor__num__imputer__strategy': ['mean','median'],
    #'preprocessor__num__scaler__with_mean': [True,False],
    #'preprocessor__num__scaler__with_std': [True,False],
}

default_scoring = 'accuracy'
