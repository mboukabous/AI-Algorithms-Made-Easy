# Dimensionality Reduction Models

This directory contains Python scripts defining **dimensionality reduction** techniques (e.g., PCA, t-SNE, UMAP). Each model file sets up a scikit-learn–compatible estimator or follows a similar interface, making it easy to swap in `train_dimred_model.py`.

**Key Points**:
- **Estimator**: Typically supports `.fit_transform(X)` for dimension reduction.
- **Default Settings**: e.g., PCA might default to `n_components=2`; t-SNE might set `n_components=2` and `perplexity=30`; UMAP might define `n_neighbors=15` or `n_components=2`.
- **No Supervised Tuning**: Usually we pick hyperparameters based on interpretability or domain. A manual approach or specialized metric can be used if needed.

**Note**: The `train_dimred_model.py` script handles dropping columns, label encoding, performing `.fit_transform(X)`, and optionally saving a 2D/3D scatter plot if `--visualize` is used.

## Available Dimensionality Reduction Models

- [PCA](pca.py)  
- [t-SNE](tsne.py)  
- [UMAP](umap.py)  

### Usage

To reduce data dimensions:

```bash
python scripts/train_dimred_model.py \
  --model_module pca \
  --data_path data/breast_cancer/data.csv \
  --select_columns "radius_mean, texture_mean, area_mean, smoothness_mean" \
  --visualize
```

This:
1. Loads `pca.py`, which defines a `PCA(n_components=2)` estimator by default.
2. Applies `.fit_transform(...)` to produce a 2D embedding.
3. Saves the model (`dimred_model.pkl`) and the transformed data (`X_transformed.csv`).
4. If `--visualize` is set and `n_components=2`, it scatter-plots the result.
