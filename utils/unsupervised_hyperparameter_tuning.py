
"""
unsupervised_hyperparameter_tuning.py

Provides a function for hyperparameter tuning of clustering models
using silhouette score as an objective.
"""

import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score
import copy

def clustering_hyperparameter_tuning(X, estimator, param_grid, scoring='silhouette', cv=5):
    """
    A simple manual hyperparameter search for clustering models,
    using silhouette_score for evaluation.

    Args:
        X (array-like): Feature data for clustering.
        estimator: An estimator with .fit() and .predict() or .labels_ attribute.
        param_grid (dict): Dictionary of hyperparams, e.g. {'model__n_clusters': [2,3,4]}.
        scoring (str): Only 'silhouette' is supported here.
        cv (int): We can do repeated subsampling or something similar to get stable silhouette.

    Returns:
        best_estimator: The estimator with best silhouette score.
        best_params: Dictionary of best parameters found.
    """
    if not param_grid:
        # If param_grid is empty, just fit once
        estimator.fit(X)
        return estimator, {}

    best_score = -1  # silhouette ranges -1 to 1
    best_params = None
    best_estimator = None

    for params in ParameterGrid(param_grid):
        # Clone the original estimator
        from sklearn.base import clone
        current_estimator = clone(estimator)

        # Apply params
        for param, val in params.items():
            # param might look like "model__n_clusters"
            # We adapt: if param starts with 'model__', we set on current_estimator
            path = param.split('__')
            if len(path) > 1:
                # E.g., path = ['model','n_clusters']
                # we set current_estimator.n_clusters = val
                setattr(current_estimator, path[1], val)
            else:
                # If there's no 'model__' prefix
                setattr(current_estimator, param, val)

        # Simple approach to do multiple splits if we want
        # For now, let's do a single fit to keep it straightforward
        current_estimator.fit(X)

        # Use the fitted current_estimator here, not 'estimator'
        if hasattr(current_estimator, 'labels_') and current_estimator.labels_ is not None:
            labels = current_estimator.labels_
        elif hasattr(current_estimator, 'predict'):
            labels = current_estimator.predict(X)
        else:
            raise ValueError("No valid way to retrieve cluster labels for this estimator.")

        unique_labels = set(labels)
        if len(unique_labels) > 1:
            score = silhouette_score(X, labels)
        else:
            score = -999  # invalid scenario if only 1 cluster

        if score > best_score:
            best_score = score
            best_params = params
            best_estimator = current_estimator

    if best_estimator is None:
        print("No valid parameter combination produced more than 1 cluster. Falling back to original estimator.")
        estimator.fit(X)
        return estimator, {}
    else:
        print(f"Best silhouette score: {best_score:.4f}")
        return best_estimator, best_params
