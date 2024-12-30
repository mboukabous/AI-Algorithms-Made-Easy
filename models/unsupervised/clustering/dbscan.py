
"""
dbscan.py

This module defines a DBSCAN clustering model and a parameter grid for hyperparameter tuning.

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm.
It groups points closely packed together and marks as outliers those points in low-density regions.

Parameters:
    - eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
    - min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered a core point.
"""

from sklearn.cluster import DBSCAN

# Define the DBSCAN estimator
estimator = DBSCAN(eps=0.5, min_samples=5)

# Define the hyperparameter grid for tuning
param_grid = {
    'model__eps': [0.2, 0.5, 1.0, 1.5, 2.0],  # Explore a wide range of neighborhood radii
    'model__min_samples': [3, 5, 10, 20]  # Adjust density thresholds for core points
}

# Default scoring metric
# Note: Silhouette score works best for convex clusters and may not always be ideal for DBSCAN.
# For more complex shapes, consider custom evaluation metrics.
default_scoring = 'silhouette'
