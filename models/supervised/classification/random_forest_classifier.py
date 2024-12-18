
"""
Random Forest Classifier setup.

Features:
- Uses `RandomForestClassifier` from scikit-learn.
- Good general-purpose model for binary and multi-class tasks.
- Default scoring: 'accuracy'.
"""

from sklearn.ensemble import RandomForestClassifier

estimator = RandomForestClassifier(random_state=42)

param_grid = {
    'model__n_estimators': [100],
    'model__max_depth': [None, 10],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1],
    # Preprocessing params
    #'preprocessor__num__imputer__strategy': ['mean', 'median'],
    #'preprocessor__num__scaler__with_mean': [True, False],
    #'preprocessor__num__scaler__with_std': [True, False],
}

default_scoring = 'accuracy'
