
"""
pca.py

This module defines a Principal Component Analysis (PCA) model for dimensionality reduction. 
PCA is a widely used technique to reduce the dimensionality of large datasets by projecting the data 
onto a lower-dimensional subspace while preserving as much variance as possible.

Key Features:
- Reduces computational complexity for high-dimensional data.
- Helps in visualizing data in 2D or 3D space.
- Useful as a preprocessing step for clustering or classification.

Parameters:
    - n_components (int, float, or None): Number of principal components to keep.
        - int: Specifies the exact number of components.
        - float: Keeps enough components to explain the specified fraction of variance (e.g., 0.95 for 95% variance).
        - None: Keeps all components (default).

Default:
    - n_components=2: Projects the data onto 2 dimensions for visualization purposes.

"""

from sklearn.decomposition import PCA

# Define the PCA estimator
estimator = PCA(n_components=2)  # Default to 2D projection for visualization
