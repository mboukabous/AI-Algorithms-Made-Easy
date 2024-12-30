
"""
gaussian_mixture.py

This module defines a GaussianMixture model for clustering, along with a parameter grid for hyperparameter tuning.

Gaussian Mixture Models (GMM) assume that data is generated from a mixture of several Gaussian distributions
with unknown parameters. It's a probabilistic model and can handle clusters of varying sizes and shapes.

Parameters:
    - n_components (int): Number of mixture components (clusters).
    - covariance_type (str): Determines the shape of each cluster.
        - 'full': Each cluster has its own general covariance matrix.
        - 'tied': All clusters share the same covariance matrix.
        - 'diag': Each cluster has its own diagonal covariance matrix.
        - 'spherical': Each cluster has its own single variance.
"""

from sklearn.mixture import GaussianMixture

# Define the GaussianMixture estimator
estimator = GaussianMixture(n_components=3, random_state=42)

# Define the hyperparameter grid for tuning
param_grid = {
    'model__n_components': [2, 3, 4],  # Experiment with 2 to 4 clusters
    'model__covariance_type': ['full', 'tied', 'diag', 'spherical']  # Different shapes for cluster covariance
}

# Default scoring metric
# Note: Silhouette score works better for convex clusters. For GMMs with non-convex clusters, consider other metrics like BIC or AIC.
default_scoring = 'silhouette'
