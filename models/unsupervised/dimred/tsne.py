
"""
tsne.py

This module defines a t-Distributed Stochastic Neighbor Embedding (t-SNE) model 
for dimensionality reduction. t-SNE is primarily used for visualizing high-dimensional 
data by projecting it into a lower-dimensional space (typically 2D or 3D).

Key Features:
- Nonlinear dimensionality reduction technique.
- Preserves local relationships within the data.
- Useful for exploring clustering structures in high-dimensional datasets.

Parameters:
    - n_components (int): Number of dimensions for projection (default: 2 for visualization).
    - perplexity (float): Controls the balance between local and global data structure.
        - Typical values range between 5 and 50.
    - learning_rate (float, optional): Learning rate for optimization (default: 'auto').
    - random_state (int, optional): Ensures reproducibility of the results.

Default:
    - n_components=2: Projects the data into a 2D space for visualization purposes.
    - perplexity=30: A good starting point for most datasets.

"""

from sklearn.manifold import TSNE

# Define the t-SNE estimator
estimator = TSNE(n_components=2, perplexity=30)  # Default to 2D projection with a reasonable perplexity
