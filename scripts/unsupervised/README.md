# Unsupervised Learning Scripts

This directory contains executable scripts for training, testing, and other tasks related to model development and evaluation.

## Contents

Unsupervised Learning:
- [train_clustering_model.py](#train_clustering_modelpy)
- [train_dimred_model.py](#train_dimred_modelpy)
- [train_anomaly_detection.py](#train_anomaly_detectionpy)

---

## `train_clustering_model.py`

A script for training **clustering** models (K-Means, DBSCAN, Gaussian Mixture, etc.) in an unsupervised manner. It supports data loading, optional drop/select of columns, label encoding for non-numeric features, optional hyperparameter tuning (silhouette-based), saving the final model, and generating a 2D cluster plot if needed.

### Features

- Supports various clustering models defined in `models/unsupervised/clustering`.
- Optional hyperparameter tuning (silhouette score) via `clustering_hyperparameter_tuning`.
- Saves the trained clustering model and optional silhouette metrics.
- Generates a 2D scatter plot if `visualize` is enabled (using PCA if needed).

### Usage

```bash
python train_clustering_model.py --model_module MODEL_MODULE \
    --data_path DATA_PATH/DATA_NAME.csv [OPTIONS]
```

**Key Arguments**:
- `model_module`: Name of the clustering model module (e.g., `kmeans`, `dbscan`, `gaussian_mixture`).
- `data_path`: Path to the CSV dataset.

**Optional Arguments**:
- `drop_columns`: Comma-separated column names to drop.
- `select_columns`: Comma-separated column names to keep.
- `tune`: If set, performs silhouette-based hyperparameter tuning.
- `cv_folds`: Number of folds or times for silhouette-based repeated runs (basic approach).
- `scoring_metric`: Typically `'silhouette'`.
- `visualize`: If set, attempts a 2D scatter, using PCA if more than 2 features remain.
- `model_path`: Path to save the trained model.
- `results_path`: Path to save results (metrics, plots).

### Usage Example

```bash
python train_clustering_model.py \
  --model_module kmeans \
  --data_path data/mall_customer/Mall_Customers.csv \
  --drop_columns "Gender" \
  --select_columns "Annual Income (k$),Spending Score (1-100)" \
  --visualize
```

---

## `train_dimred_model.py`

A script for **dimensionality reduction** tasks (e.g., PCA, t-SNE, UMAP). It loads data, optionally drops or selects columns, label-encodes categorical features, fits the chosen dimensionality reduction model, saves the transformed data, and can visualize 2D/3D outputs.

### Features

- Supports various dimension reduction models in `models/unsupervised/dimred`.
- Saves the fitted model and the transformed data (in CSV).
- Optionally creates a 2D or 3D scatter plot if the output dimension is 2 or 3.

### Usage

```bash
python train_dimred_model.py --model_module MODEL_MODULE \
    --data_path DATA_PATH/DATA_NAME.csv [OPTIONS]
```

**Key Arguments**:
- `model_module`: Name of the dimension reduction module (e.g., `pca`, `tsne`, `umap`).
- `data_path`: Path to the CSV dataset.

**Optional Arguments**:
- `drop_columns`: Comma-separated column names to drop.
- `select_columns`: Comma-separated column names to keep.
- `visualize`: If set, plots the 2D or 3D embedding.
- `model_path`: Path to save the trained model.
- `results_path`: Path to save the transformed data and any plots.

### Usage Example

```bash
python train_dimred_model.py \
  --model_module pca \
  --data_path data/breast_cancer/data.csv \
  --drop_columns "id,diagnosis" \
  --visualize
```

---

## `train_anomaly_detection.py`

A script for training **anomaly/outlier detection** models (Isolation Forest, One-Class SVM, etc.). It supports dropping/selecting columns, label-encoding, saving anomaly predictions (0 = normal, 1 = outlier), and optionally visualizing points in 2D with outliers colored differently.

### Features

- Supports various anomaly models in `models/unsupervised/anomaly`.
- Saves the model and an outlier predictions CSV.
- If `visualize` is enabled, performs PCA → 2D for plotting normal vs. outliers.

### Usage

```bash
python train_anomaly_detection.py --model_module MODEL_MODULE \
    --data_path DATA_PATH/DATA_NAME.csv [OPTIONS]
```

**Key Arguments**:
- `model_module`: Name of the anomaly detection module (e.g., `isolation_forest`, `one_class_svm`, `local_outlier_factor`).
- `data_path`: Path to the CSV dataset.

**Optional Arguments**:
- `drop_columns`: Comma-separated column names to drop.
- `select_columns`: Comma-separated column names to keep.
- `visualize`: If set, attempts a 2D scatter (via PCA) and colors outliers in red.
- `model_path`: Path to save the anomaly model.
- `results_path`: Path to save outlier predictions and plots.

### Usage Example

```bash
python train_anomaly_detection.py \
  --model_module isolation_forest \
  --data_path data/breast_cancer/data.csv \
  --drop_columns "id,diagnosis" \
  --visualize
```
