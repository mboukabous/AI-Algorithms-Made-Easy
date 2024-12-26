
"""
local_outlier_factor.py

This module defines a Local Outlier Factor (LOF) model for anomaly detection. 
LOF identifies anomalies by comparing the local density of a sample to the density 
of its neighbors. Samples with significantly lower density are flagged as outliers.

Key Features:
- Detects local anomalies in datasets with varying densities.
- Effective for datasets where the notion of an outlier is context-dependent.
- Non-parametric method that adapts to the data's structure.

Parameters:
    - n_neighbors (int): Number of neighbors used to calculate local density.
        - Default: 20. Higher values smooth out anomalies but may miss local patterns.
    - contamination (str or float): Proportion of outliers in the data.
        - 'auto': Automatically estimates the proportion based on the dataset size.
        - float: Manually set the expected proportion (e.g., 0.1 for 10%).
    - novelty (bool): If True, allows the model to be applied to new unseen data.

Limitations:
- LOF directly computes predictions during `fit_predict()` and does not support `predict()` 
  unless `novelty=True`.

Default Configuration:
    - n_neighbors=20: Uses 20 neighbors for density comparison.
    - contamination='auto': Automatically estimates the proportion of outliers.
    - novelty=True: Enables predictions on unseen data.
"""

from sklearn.neighbors import LocalOutlierFactor

# Define the Local Outlier Factor estimator
estimator = LocalOutlierFactor(
    n_neighbors=20,  # Number of neighbors to calculate density
    contamination='auto',  # Auto-detect the proportion of outliers
    novelty=True  # Enables prediction on new data
)
