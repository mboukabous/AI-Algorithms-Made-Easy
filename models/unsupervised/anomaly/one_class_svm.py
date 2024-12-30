
"""
one_class_svm.py

This module defines a One-Class SVM model for anomaly detection. 
One-Class SVM identifies a decision boundary that separates normal data points from potential outliers 
in a high-dimensional feature space.

Key Features:
- Effective for detecting anomalies in high-dimensional datasets.
- Flexible kernel options for nonlinear decision boundaries.
- Suitable for datasets with a small proportion of outliers.

Parameters:
    - kernel (str): Specifies the kernel type used in the algorithm.
        - Common options: 'linear', 'poly', 'rbf' (default), and 'sigmoid'.
    - gamma (str or float): Kernel coefficient. Determines the influence of each sample.
        - Default: 'scale' (1 / (n_features * X.var())).
    - nu (float): Approximate fraction of outliers in the dataset.
        - Must be in the range (0, 1]. Default: 0.05 (5% of data considered outliers).

Default Configuration:
    - kernel='rbf': Radial Basis Function for nonlinear separation.
    - gamma='scale': Automatically adjusts kernel influence based on dataset features.
    - nu=0.05: Assumes approximately 5% of data points are outliers.
"""

from sklearn.svm import OneClassSVM

# Define the One-Class SVM estimator
estimator = OneClassSVM(
    kernel='rbf',  # Radial Basis Function kernel for nonlinear boundaries
    gamma='scale',  # Adjusts kernel influence based on dataset variance
    nu=0.05         # Assumes 5% of the data are outliers
)
