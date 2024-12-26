
"""
kmeans.py

This module defines a KMeans clustering model and a parameter grid for hyperparameter tuning.

KMeans is a popular clustering algorithm that partitions data into k clusters. Each cluster is represented by the centroid of its members, and the algorithm iteratively refines the centroids to minimize the within-cluster variance.

Parameters:
    - n_clusters (int): Number of clusters to form.
    - init (str): Initialization method for centroids. Common options:
        - 'k-means++' (default): Optimized centroid initialization.
        - 'random': Random initialization.
    - n_init (int): Number of times the algorithm runs with different centroid seeds.
    - random_state (int): Ensures reproducibility of results.
"""

from sklearn.cluster import KMeans

# Define the KMeans estimator
estimator = KMeans(n_clusters=3, random_state=42)

# Define the hyperparameter grid for tuning
param_grid = {
    'model__n_clusters': [2, 3, 4, 5],  # Experiment with 2 to 5 clusters
    'model__init': ['k-means++', 'random'],  # Compare optimized and random initialization
    'model__n_init': [10, 20, 50]  # Test different numbers of initializations for stability
}

# Use silhouette score as the default scoring metric
# Silhouette score evaluates how well clusters are separated and compact
default_scoring = 'silhouette'
