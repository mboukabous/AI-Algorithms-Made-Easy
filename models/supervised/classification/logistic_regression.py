
"""
This module sets up a Logistic Regression classifier for hyperparameter tuning.

Features:
- Uses `LogisticRegression` from scikit-learn.
- Defines a hyperparameter grid for both preprocessing and model parameters.
- Suitable for binary and multi-class classification (LogisticRegression uses OvR/One-vs-Rest by default).
- Default scoring: 'accuracy', which works well for both binary and multi-class tasks.

Considerations:
- Adjusting `C` controls regularization strength.
- `penalty='l2'` is commonly used.
- One can add more solvers or penalties as needed.
"""

from sklearn.linear_model import LogisticRegression

# Define the estimator
estimator = LogisticRegression()

# Define the hyperparameter grid
param_grid = {
    # Model parameters
    'model__C': [0.01, 0.1, 1.0, 10.0],  # Regularization strength
    'model__penalty': ['l2'],            # Only L2 regularization supported in LogisticRegression(solver='lbfgs')
    'model__solver': ['lbfgs'],  # Efficient solver for large datasets
    'model__max_iter': [1000] # Control convergence
    # Preprocessing parameters for numerical features
    #'preprocessor__num__imputer__strategy': ['mean', 'median'],
    #'preprocessor__num__scaler__with_mean': [True, False],
    #'preprocessor__num__scaler__with_std': [True, False],
}

# Optional: Default scoring metric for classification
default_scoring = 'accuracy'
