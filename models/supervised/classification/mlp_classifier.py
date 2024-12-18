
"""
MLP Classifier setup.

Features:
- Uses `MLPClassifier`.
- Suitable for binary and multi-class classification.
- Default scoring: 'accuracy'.

Considerations:
- `hidden_layer_sizes`, `alpha` (L2 regularization), and `learning_rate_init` are common parameters.
- Increase `max_iter` if convergence warnings appear.
"""

from sklearn.neural_network import MLPClassifier

# Define the estimator
estimator = MLPClassifier(max_iter=200, random_state=42)

# Define the hyperparameter grid
param_grid = {
    'model__hidden_layer_sizes': [(50,)],  # Reduced size of hidden layers for faster training
    'model__alpha': [0.001],  # Retained commonly effective value
    'model__learning_rate_init': [0.001],  # Focused on a single typical value for faster tuning
    # Uncomment and customize preprocessing params if needed
    #'preprocessor__num__imputer__strategy': ['mean'],
    #'preprocessor__num__scaler__with_mean': [True],
    #'preprocessor__num__scaler__with_std': [True],
}

default_scoring = 'accuracy'
