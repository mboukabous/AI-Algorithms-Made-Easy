# Anomaly (Outlier) Detection Models

This directory hosts scripts defining **anomaly detection** estimators (e.g., Isolation Forest, One-Class SVM, etc.) for use with `train_anomaly_detection.py`. Each file specifies a scikit-learn–compatible outlier detector and, if applicable, a parameter grid.

**Key Points**:
- **Estimator**: Must allow `.fit(X)` and `.predict(X)` or similar. Typically returns +1 / −1 for inliers / outliers (we unify to 0 / 1).
- **Parameter Grid**: You can define hyperparameters (like `n_estimators`, `contamination`) for potential searching. 
- **Default Approach**: We do not rely on labeled anomalies (unsupervised). The script will produce a predictions CSV with 0 = normal, 1 = outlier.

**Note**: The main script `train_anomaly_detection.py` handles data loading, label encoding, dropping/selecting columns, the `.fit(X)`, `.predict(X)` steps, saving the outlier predictions, and (optionally) a 2D plot with outliers in red.

## Available Anomaly Detection Models

- [Isolation Forest](isolation_forest.py)  
- [One-Class SVM](one_class_svm.py)  
- [Local Outlier Factor (LOF)](local_outlier_factor.py)  

### Usage

For example, to detect outliers with an Isolation Forest:

```bash
python scripts/train_anomaly_detection.py \
  --model_module isolation_forest \
  --data_path data/breast_cancer/data.csv \
  --drop_columns "id,diagnosis" \
  --visualize
```

This:
1. Loads `isolation_forest.py`, sets up `IsolationForest(...)`.
2. Fits the model to the data, saves it, then `predict(...)`.
3. Saves a `predictions.csv` with `OutlierPrediction`.
4. If `--visualize`, does a 2D PCA scatter, coloring outliers red.
