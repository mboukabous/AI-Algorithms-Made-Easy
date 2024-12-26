# Scripts

This directory contains executable scripts for training, testing, and other tasks related to model development and evaluation.

## Contents

Supervised Learning:
- [train_regression_model.py](#train_regression_modelpy)
- [train_classification_model.py](#train_classification_modelpy)

Unsupervised Learning:
- [train_clustering_model.py](#train_clustering_modelpy)
- [train_dimred_model.py](#train_dimred_modelpy)
- [train_anomaly_detection.py](#train_anomaly_detectionpy)

---

## `train_regression_model.py`

A script for training supervised learning **regression** models using scikit-learn. It handles data loading, preprocessing, optional log transformation, hyperparameter tuning, model evaluation, and saving of models, metrics, and visualizations.

### Features

- Supports various regression models defined in `models/supervised/regression`.
- Performs hyperparameter tuning using grid search cross-validation.
- Saves trained models and evaluation metrics.
- Generates visualizations if specified.

### Usage

```bash
python train_regression_model.py --model_module MODEL_MODULE \
    --data_path DATA_PATH/DATA_NAME.csv \
    --target_variable TARGET_VARIABLE [OPTIONS]
```

**Required Arguments**:
- `model_module`: Name of the regression model module to import (e.g., `linear_regression`).
- `data_path`: Path to the dataset directory, including the data file name.
- `target_variable`: Name of the target variable.

**Optional Arguments**:
- `test_size`: Proportion of the dataset to include in the test split (default: `0.2`).
- `random_state`: Random seed for reproducibility (default: `42`).
- `log_transform`: Apply log transformation to the target variable (regression only).
- `cv_folds`: Number of cross-validation folds (default: `5`).
- `scoring_metric`: Scoring metric for model evaluation.
- `model_path`: Path to save the trained model.
- `results_path`: Path to save results and metrics.
- `visualize`: Generate and save visualizations (e.g., scatter or actual vs. predicted).
- `drop_columns`: Comma-separated column names to drop from the dataset.

### Usage Example

```bash
python train_regression_model.py --model_module linear_regression \
    --data_path data/house_prices/train.csv \
    --target_variable SalePrice --drop_columns Id \
    --log_transform --visualize
```

---

## `train_classification_model.py`

A script for training supervised learning **classification** models using scikit-learn. It handles data loading, preprocessing, hyperparameter tuning (via grid search CV), model evaluation using classification metrics, and saving of models, metrics, and visualizations.

### Features

- Supports various classification models defined in `models/supervised/classification`.
- Performs hyperparameter tuning using grid search cross-validation (via `classification_hyperparameter_tuning`).
- Saves trained models and evaluation metrics (accuracy, precision, recall, F1).
- If `visualize` is enabled, it generates a metrics bar chart and a confusion matrix plot.

### Usage

```bash
python train_classification_model.py --model_module MODEL_MODULE \
    --data_path DATA_PATH/DATA_NAME.csv \
    --target_variable TARGET_VARIABLE [OPTIONS]
```

**Required Arguments**:
- `model_module`: Name of the classification model module to import (e.g., `logistic_regression`).
- `data_path`: Path to the dataset directory, including the data file name.
- `target_variable`: Name of the target variable (categorical).

**Optional Arguments**:
- `test_size`: Proportion of the dataset to include in the test split (default: `0.2`).
- `random_state`: Random seed for reproducibility (default: `42`).
- `cv_folds`: Number of cross-validation folds (default: `5`).
- `scoring_metric`: Scoring metric for model evaluation (e.g., `accuracy`, `f1`, `roc_auc`).
- `model_path`: Path to save the trained model.
- `results_path`: Path to save results and metrics.
- `visualize`: Generate and save visualizations (metrics bar chart, confusion matrix).
- `drop_columns`: Comma-separated column names to drop from the dataset.

### Usage Example

```bash
python train_classification_model.py --model_module logistic_regression \
    --data_path data/adult_income/train.csv \
    --target_variable income_bracket \
    --scoring_metric accuracy --visualize
```

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
