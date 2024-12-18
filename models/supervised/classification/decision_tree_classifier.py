
"""
This module sets up a Decision Tree Classifier for hyperparameter tuning.

Features:
- Uses `DecisionTreeClassifier` from scikit-learn.
- Defines a parameter grid suitable for both binary and multi-class classification.
- Default scoring: 'accuracy'.

Considerations:
- `criterion`, `max_depth`, `min_samples_split`, and `min_samples_leaf` are common parameters to tune.
- Ordinal encoding will be used for tree-based models if implemented, but the pipeline code decides that.

"""

from sklearn.tree import DecisionTreeClassifier

estimator = DecisionTreeClassifier(random_state=42)

param_grid = {
    'model__criterion': ['gini', 'entropy'],
    'model__max_depth': [None, 5, 10],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2],
    # Preprocessing params
    #'preprocessor__num__imputer__strategy': ['mean', 'median'],
    #'preprocessor__num__scaler__with_mean': [True, False],
    #'preprocessor__num__scaler__with_std': [True, False],
}

default_scoring = 'accuracy'
