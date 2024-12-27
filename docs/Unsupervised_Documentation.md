# Unsupervised Learning Models Documentation

## 1. Introduction

Welcome to the **Unsupervised Learning Models Documentation**! This guide provides detailed instructions on how to leverage unsupervised algorithms implemented in Python with scikit-learn and related libraries. The scripts are designed to be modular and reusable, enabling you to **cluster**, **reduce dimensions**, and **detect anomalies** in your datasets.

**Key Features:**

- **Modular design** with separate scripts for each unsupervised task (clustering, dimensionality reduction, anomaly detection).
- **Hyperparameter tuning** for clustering (silhouette-based) and optional approach for others.
- **Preprocessing pipelines** for numerical/categorical data (label encoding, dropping columns, etc.).
- **Visualization** of clusters, outliers, or dimension-reduced spaces (2D, 3D).
- **Utility scripts** for data download and preparation.

---

## 2. Installation and Setup

### Prerequisites

- **Python 3.7 or higher**
- **pip** package manager
- **Kaggle API credentials** (`kaggle.json`) if you plan to download datasets from Kaggle.

### Required Libraries

Install the necessary libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**Example `requirements.txt`:**

```
pandas==2.2.2
numpy==1.26.4
matplotlib==3.8.0
seaborn==0.13.2
kaggle==1.6.17
scikit-learn==1.5.2
catboost==1.2.7
dask[dataframe]==2024.10.0
xgboost==2.1.2
lightgbm==4.5.0
joblib==1.4.2
gradio==5.7.1
umap-learn==0.5.7
```

### Project Structure

Your project directory might be organized like this:

```
project_root/
├── data/
│   ├── raw/
│   │   └── your_dataset/
│   │       └── data.csv
│   └── datasets/
│       └── kaggle_data.py
├── interfaces/
│   └── gradio/
│       └── train_unsupervised_gradio.py
├── models/
│   └── unsupervised/
│       ├── clustering/
│       │   ├── kmeans.py
│       │   ├── dbscan.py
│       │   ├── gaussian_mixture.py
│       │   └── hierarchical_clustering.py
│       ├── dimred/
│       │   ├── pca.py
│       │   ├── tsne.py
│       │   └── umap.py
│       └── anomaly/
│           ├── isolation_forest.py
│           ├── one_class_svm.py
│           └── local_outlier_factor.py
├── scripts/
│   ├── train_clustering_model.py
│   ├── train_dimred_model.py
│   └── train_anomaly_detection.py
├── utils/
│   └── unsupervised_hyperparameter_tuning.py
├── saved_models/
├── results/
├── requirements.txt
└── README.md
```

---

## 3. Data Preparation

### Dataset Requirements

- **CSV format** is recommended.
- **Numeric columns** or easily label-encoded categorical columns.
- If your dataset has completely empty columns or all-NaN columns, remove them or fix them before modeling.
- Large datasets can demand significant CPU/RAM, especially for t-SNE/UMAP or advanced clustering.

### Example Datasets

We often demonstrate:

- **Mall Customers** dataset for **Clustering**.  
- **Breast Cancer Wisconsin** (numeric features) for **Dimensionality Reduction** or **Anomaly Detection**.  
- Any dataset with potential outliers or interesting group structures for these tasks.

### Downloading from Kaggle

If you want to fetch a dataset from Kaggle:

```python
from data.datasets.kaggle_data import get_kaggle_data

JSON_KAGGLE_PATH = "/path/to/kaggle.json"
DATA_NAME = "vjchoudhary7/customer-segmentation-tutorial-in-python"
is_competition = False

dataset_path = get_kaggle_data(JSON_KAGGLE_PATH, DATA_NAME, is_competition)
print(f"Dataset available at: {dataset_path}")
```

---

## 4. Clustering

Clustering aims to group data points without labels. The main script is `scripts/train_clustering_model.py`, which:

- **Loads** CSV data, optionally dropping or selecting columns.
- **Label-encodes** any non-numeric columns.
- **Optionally** does silhouette-based hyperparameter tuning (e.g., for `KMeans`, `DBSCAN`).
- **Fits** the final model, saves it, and (optionally) plots 2D clusters via PCA.

### Running the Clustering Script

```bash
python scripts/train_clustering_model.py \
  --model_module kmeans \
  --data_path data/mall_customer/Mall_Customers.csv \
  --drop_columns "Gender" \
  --select_columns "Annual Income (k$),Spending Score (1-100)" \
  --tune \
  --visualize
```

**Key Arguments**:
- `--model_module`: Clustering model name (e.g., `kmeans`, `dbscan`, `gaussian_mixture`).
- `--tune`: Enable silhouette-based hyperparameter tuning.
- `--cv_folds`: If you want repeated runs or advanced cross-checking (basic approach uses single fits).
- `--visualize`: Creates a 2D scatter (using PCA if needed) coloring each cluster differently.

### Example: K-Means

```bash
python scripts/train_clustering_model.py \
  --model_module kmeans \
  --data_path data/mall_customer/Mall_Customers.csv \
  --visualize
```
This will load `kmeans.py`, potentially read its `param_grid` for `n_clusters`, etc., run or skip tuning, produce cluster assignments, and save the model.

---

## 5. Dimensionality Reduction

Dimensionality reduction transforms high-dimensional data into fewer features (2D, 3D) for visualization or further analysis. The main script is `scripts/train_dimred_model.py`.

### Running the Dimensionality Reduction Script

```bash
python scripts/train_dimred_model.py \
  --model_module pca \
  --data_path data/breast_cancer/data.csv \
  --drop_columns "id,diagnosis" \
  --visualize
```

**Key Arguments**:
- `--model_module`: e.g. `pca`, `tsne`, `umap`.
- `--visualize`: If 2D or 3D, saves a scatter plot of the transformed data.

#### PCA Example

```bash
python scripts/train_dimred_model.py \
  --model_module pca \
  --data_path data/breast_cancer/data.csv \
  --select_columns "radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean" \
  --visualize
```

This:
1. Loads `pca.py` (which might define `PCA(n_components=2)`).
2. Encodes any categorical columns, drops empty columns if needed.
3. Applies `.fit_transform(X)`, saves `dimred_model.pkl` and the transformed data (`X_transformed.csv`).
4. If `n_components=2`, generates a scatter plot.

---

## 6. Anomaly (Outlier) Detection

Anomaly detection identifies unusual points that deviate from most observations. The main script is `scripts/train_anomaly_detection.py`.

### Running the Anomaly Detection Script

```bash
python scripts/train_anomaly_detection.py \
  --model_module isolation_forest \
  --data_path data/breast_cancer/data.csv \
  --drop_columns "id,diagnosis" \
  --visualize
```

**Key Arguments**:
- `--model_module`: e.g. `isolation_forest`, `one_class_svm`, `local_outlier_factor`.
- `--visualize`: Creates a 2D PCA scatter, marking outliers in red vs. normal points in blue.

#### Example: Isolation Forest

```bash
python scripts/train_anomaly_detection.py \
  --model_module isolation_forest \
  --data_path data/breast_cancer/data.csv \
  --drop_columns "id,diagnosis" \
  --visualize
```

1. Loads `isolation_forest.py` (with `IsolationForest`).
2. Fits the model on all data.
3. Predicts outliers (1) vs. normal (0) in `predictions.csv`.
4. If `visualize`, does PCA→2D scatter with color-coded outliers.

---

## 7. Usage Examples

### Clustering with DBSCAN

```bash
python scripts/train_clustering_model.py \
  --model_module dbscan \
  --data_path data/mall_customer/Mall_Customers.csv \
  --tune \
  --visualize
```
- Searches hyperparameters (like `eps`, `min_samples`) via silhouette.  
- Saves final DBSCAN model, cluster labels, and optional 2D cluster plot.

### Dimensionality Reduction with UMAP

```bash
python scripts/train_dimred_model.py \
  --model_module umap \
  --data_path data/wholesale_customers/Wholesale_customers_data.csv \
  --visualize
```
- Runs UMAP with `n_components=2`, saves `umap_model.pkl`, and a 2D scatter of the reduced data.

### Anomaly Detection with One-Class SVM

```bash
python scripts/train_anomaly_detection.py \
  --model_module one_class_svm \
  --data_path data/credit_card/creditcard.csv \
  --visualize
```
- Identifies outliers (potential fraud) using One-Class SVM, outputs a 2D PCA scatter if more than 2 features.

---

## 8. Disclaimers and Notes

### Large Datasets

- Clustering (DBSCAN, Agglomerative) or advanced methods (t-SNE, UMAP) can be **RAM/CPU intensive**.  
- For extremely large data, consider sampling or approximate algorithms.

### Missing/NaN Values

- Ensure columns with all NaN are dropped automatically or fixed.  
- Impute or drop partial NaNs. Most unsupervised methods require numeric, non-empty data.

### Overlapping or Single Clusters

- Some algorithms (DBSCAN) might yield a single cluster if parameters are not well-chosen. Silhouette-based tuning helps, but not always guaranteed to find multiple clusters if the data truly doesn’t cluster well.

### Random Seeds

- **Set** `--random_state 42` or an appropriate seed for reproducibility in certain algorithms like K-Means, PCA, or UMAP if you want consistent runs.

---

## 9. Conclusion

This documentation outlines how to run **unsupervised learning** tasks—**clustering**, **dimensionality reduction**, and **anomaly detection**—using the provided scripts:

- **`train_clustering_model.py`**: For grouping data without labels.  
- **`train_dimred_model.py`**: For compressing high-dimensional data into 2D/3D for insight or further analysis.  
- **`train_anomaly_detection.py`**: For finding outliers in unlabeled datasets.

Each script is flexible in terms of hyperparameters, data preprocessing, and optional visualization. By following these guidelines, you can:

1. Choose or prepare your dataset (handle missing values, label-encode, etc.).  
2. Select the best-suited unsupervised algorithm or model module.  
3. Optionally do hyperparameter tuning (like silhouette for clustering).  
4. Inspect your results (clusters, dimension-reduced plots, or outlier markings).

We encourage you to experiment with different datasets and hyperparameters to find the approach that offers the most insight into your data.

---

## Additional Resources

- **scikit-learn Clustering**: [https://scikit-learn.org/stable/modules/clustering.html](https://scikit-learn.org/stable/modules/clustering.html)
- **scikit-learn Manifold Learning** (t-SNE, etc.): [https://scikit-learn.org/stable/modules/manifold.html](https://scikit-learn.org/stable/modules/manifold.html)
- **UMAP**: [https://umap-learn.readthedocs.io/en/latest/](https://umap-learn.readthedocs.io/en/latest/)
- **scikit-learn Outlier Detection**: [https://scikit-learn.org/stable/modules/outlier_detection.html](https://scikit-learn.org/stable/modules/outlier_detection.html)
- **Kaggle API**: [https://www.kaggle.com/docs/api](https://www.kaggle.com/docs/api)

---

## Contact

If you have any questions or need further assistance, please don’t hesitate to reach out or open an issue. Enjoy exploring your data through unsupervised learning!
