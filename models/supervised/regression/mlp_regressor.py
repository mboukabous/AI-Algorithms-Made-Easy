
"""
This module sets up a Multilayer Perceptron Regressor with hyperparameter tuning.

Features:
- Uses `MLPRegressor` estimator from scikit-learn.
- Defines a hyperparameter grid for neural network parameters.
- Capable of capturing complex non-linear relationships.

Special Considerations:
- Feature scaling is crucial for MLP.
- May produce convergence warnings; increase `max_iter` to address this.
- Can be sensitive to hyperparameter settings; tuning is important.
"""

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Define the estimator
estimator = MLPRegressor(random_state=42, max_iter=1000)

# Define the hyperparameter grid
param_grid = {
    'model__hidden_layer_sizes': [(50,), (100,), (50, 50)],  # Simplified layer sizes
    'model__activation': ['relu'],  # Focused on ReLU, the most commonly effective activation
    'model__solver': ['adam'],  # Retain 'adam' for efficiency; drop 'lbfgs' (slower for larger datasets)
    'model__alpha': [0.0001, 0.001],  # Regularization strengths
    'model__learning_rate': ['constant', 'adaptive'],  # Common learning rate strategies
    'preprocessor__num__imputer__strategy': ['mean'],  # Single imputation strategy
    'preprocessor__num__scaler__with_mean': [True],  # StandardScaler
    'preprocessor__num__scaler__with_std': [True],  # StandardScaler
}

# Optional: Default scoring metric
default_scoring = 'neg_root_mean_squared_error'
