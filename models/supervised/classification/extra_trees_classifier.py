
"""
Extra Trees Classifier setup.

Features:
- Uses `ExtraTreesClassifier`.
- Similar to RandomForest but with more randomness in splits.
- Works well for both binary and multi-class.
"""

from sklearn.ensemble import ExtraTreesClassifier

estimator = ExtraTreesClassifier(random_state=42)

param_grid = {
    'model__n_estimators': [100],
    'model__max_depth': [None, 10],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1],
    # Preprocessing params
    #'preprocessor__num__imputer__strategy': ['mean','median'],
    #'preprocessor__num__scaler__with_mean': [True,False],
    #'preprocessor__num__scaler__with_std': [True,False],
}

default_scoring = 'accuracy'
