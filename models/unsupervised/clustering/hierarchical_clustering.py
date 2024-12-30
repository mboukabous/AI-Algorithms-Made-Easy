
"""
hierarchical_clustering.py

This module defines an AgglomerativeClustering model for hierarchical clustering, 
along with a parameter grid for hyperparameter tuning.

Hierarchical clustering creates a tree-like structure (dendrogram) to represent the nested grouping of data points 
and their similarity levels. Agglomerative clustering starts with each data point as its own cluster and iteratively merges them.

Parameters:
    - n_clusters (int): The number of clusters to form.
    - linkage (str): Determines how distances between clusters are computed.
        - 'ward': Minimizes the variance of clusters (requires Euclidean distance).
        - 'complete': Maximum linkage, i.e., uses the farthest points between clusters.
        - 'average': Average linkage, i.e., uses the mean distances between clusters.
        - 'single': Minimum linkage, i.e., uses the closest points between clusters.
"""

from sklearn.cluster import AgglomerativeClustering

# Define the AgglomerativeClustering estimator
estimator = AgglomerativeClustering(n_clusters=3)

# Define the hyperparameter grid for tuning
param_grid = {
    'model__n_clusters': [2, 3, 4],  # Experiment with 2 to 4 clusters
    'model__linkage': ['ward', 'complete', 'average', 'single']  # Different linkage methods for clustering
}

# Default scoring metric
# Note: Silhouette score works well for evaluating convex clusters formed by hierarchical clustering.
default_scoring = 'silhouette'
