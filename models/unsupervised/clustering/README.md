# Clustering Models

This directory contains Python scripts defining various **clustering** models and their associated hyperparameter grids. Each model file sets up a scikit-learn–compatible clustering estimator (e.g., `KMeans`, `DBSCAN`, `GaussianMixture`) and defines a param grid for the `train_clustering_model.py` script.

**Key Points**:
- **Estimator**: Usually supports `.fit(X)` for unsupervised training, and either `.labels_` or `.predict(X)` to retrieve cluster assignments.
- **Parameter Grid (`param_grid`)**: Used for silhouette-based hyperparameter tuning in `train_clustering_model.py`.
- **Default Scoring**: Often `'silhouette'`, but can be changed if you adapt your tuning logic.

**Note**: Preprocessing (dropping columns, label encoding) and any hyperparameter loop is handled externally by the script/utility. These model definition files simply define:
- An **estimator** (like `KMeans(n_clusters=3, random_state=42)`).
- A **`param_grid`** for silhouette tuning (e.g., `{'model__n_clusters':[2,3,4]}`).
- Optionally, a **`default_scoring`** set to `'silhouette'`.

## Available Clustering Models

- [KMeans](kmeans.py)  
- [DBSCAN](dbscan.py)  
- [Gaussian Mixture](gaussian_mixture.py)  
- [Agglomerative Clustering (Hierarchical)](hierarchical_clustering.py)  )

### Usage

To train or tune any clustering model, specify the `--model_module` argument with the appropriate model name (e.g., `kmeans`) when running `train_clustering_model.py`, for example:

```bash
python scripts/train_clustering_model.py \
  --model_module kmeans \
  --data_path data/mall_customer/Mall_Customers.csv \
  --tune \
  --visualize
```

This will:
1. Load the chosen model definition (`kmeans.py`).
2. Perform optional silhouette-based hyperparameter tuning if `--tune` is used.
3. Fit the final model, save it, and optionally generate a 2D scatter plot if requested.
