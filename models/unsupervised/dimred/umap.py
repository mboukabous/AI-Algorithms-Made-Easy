
"""
umap.py

This module defines a Uniform Manifold Approximation and Projection (UMAP) model 
for dimensionality reduction. UMAP is a nonlinear dimensionality reduction technique 
that is efficient for visualizing and analyzing high-dimensional data.

Key Features:
- Preserves both local and global data structures better than t-SNE in some cases.
- Scales efficiently to larger datasets compared to t-SNE.
- Suitable for exploratory data analysis and clustering.

Parameters:
    - n_components (int): Number of dimensions for projection (default: 2 for visualization).
    - n_neighbors (int): Determines the size of the local neighborhood to consider for manifold approximation. 
        - Typical values range between 5 and 50.
    - min_dist (float): Minimum distance between points in the low-dimensional space.
        - Smaller values maintain tighter clusters.
    - metric (str): Distance metric for computing similarity (default: 'euclidean').

Default:
    - n_components=2: Projects the data into a 2D space for visualization purposes.
    - n_neighbors=15: Balances local and global structure preservation.
    - min_dist=0.1: Provides moderate clustering while preserving distances.

Requirements:
    - umap-learn library must be installed.
"""

# Import UMAP from the umap-learn library
import umap.umap_ as umap

# Define the UMAP estimator
estimator = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)  # Default configuration for 2D projection
