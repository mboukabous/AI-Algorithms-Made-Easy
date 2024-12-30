# Utils

This directory contains utility scripts and helper functions that are used throughout the project. These scripts provide common functionalities such as data preprocessing, hyperparameter tuning, and other support functions that assist in model training and evaluation for **supervised** (regression and classification) as well as **unsupervised** (clustering) tasks.

## Contents

- [supervised_hyperparameter_tuning.py](#supervised_hyperparameter_tuningpy)
- [unsupervised_hyperparameter_tuning.py](#unsupervised_hyperparameter_tuningpy)

---

## `supervised_hyperparameter_tuning.py`

This script contains functions for performing hyperparameter tuning on **supervised learning** models (both regression and classification) using scikit-learn's `Pipeline` and `GridSearchCV`.

### Functions

#### `regression_hyperparameter_tuning(X, y, estimator, param_grid, cv=5, scoring=None)`

Performs hyperparameter tuning for **regression** models.

- **Parameters**:
  - `X (pd.DataFrame)`: Feature matrix.
  - `y (pd.Series)`: Numeric target variable.
  - `estimator`: A scikit-learn regressor (e.g., `LinearRegression()`).
  - `param_grid (dict)`: Parameter names and lists of values (e.g. `{'model__fit_intercept': [True, False]}`).
  - `cv (int)`: Number of cross-validation folds (default 5).
  - `scoring (str)`: Scoring metric (e.g., `'neg_root_mean_squared_error'`).
- **Returns**:
  - `best_model`: The pipeline with the best hyperparameters.
  - `best_params (dict)`: The dictionary of best hyperparameters.

**Example**:

```python
from utils.supervised_hyperparameter_tuning import regression_hyperparameter_tuning
from sklearn.linear_model import LinearRegression

X = ...  # Your regression features
y = ...  # Your numeric target variable
param_grid = {
    'model__fit_intercept': [True, False]
}

best_model, best_params = regression_hyperparameter_tuning(
    X, y, LinearRegression(), param_grid, scoring='neg_root_mean_squared_error'
)
```

---

#### `classification_hyperparameter_tuning(X, y, estimator, param_grid, cv=5, scoring=None)`

Performs hyperparameter tuning for **classification** models.

- **Parameters**:
  - `X (pd.DataFrame)`: Feature matrix.
  - `y (pd.Series)`: Target variable (binary or multi-class).
  - `estimator`: A scikit-learn classifier (e.g., `LogisticRegression()`, `RandomForestClassifier()`).
  - `param_grid (dict)`: Parameter names and lists of values (e.g. `{'model__n_estimators': [100, 200]}`).
  - `cv (int)`: Number of cross-validation folds (default 5).
  - `scoring (str)`: Scoring metric (e.g., `'accuracy'`, `'f1'`, `'roc_auc'`).
- **Returns**:
  - `best_model`: The pipeline with the best hyperparameters.
  - `best_params (dict)`: The dictionary of best hyperparameters.

**Example**:

```python
from utils.supervised_hyperparameter_tuning import classification_hyperparameter_tuning
from sklearn.ensemble import RandomForestClassifier

X = ...  # Your classification features
y = ...  # Binary or multi-class labels
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10]
}

best_model, best_params = classification_hyperparameter_tuning(
    X, y, RandomForestClassifier(), param_grid, scoring='accuracy'
)
```

---

## `unsupervised_hyperparameter_tuning.py`

This script provides a function for **hyperparameter tuning of clustering models** using **silhouette score** as the objective metric. Unlike supervised approaches, clustering does not have labeled data, so the silhouette score is used to measure how well-separated the clusters are.

### Functions

#### `clustering_hyperparameter_tuning(X, estimator, param_grid, scoring='silhouette', cv=5)`

A simple manual hyperparameter search for clustering models.

- **Parameters**:
  - `X (array-like)`: Feature matrix for clustering.
  - `estimator`: A scikit-learn clustering estimator supporting `.fit(X)` and either `.labels_` or `.predict(X)` (e.g., `KMeans`, `DBSCAN`, `GaussianMixture`).
  - `param_grid (dict)`: Dictionary of hyperparams (e.g., `{'model__n_clusters': [2,3,4]}`).
  - `scoring (str)`: Only `'silhouette'` is supported.  
  - `cv (int)`: Optionally, you could do repeated subsampling or advanced logic for more stable estimates, but the default implementation does a single fit.
- **Returns**:
  - `best_estimator`: The fitted estimator with the best silhouette score.
  - `best_params (dict)`: The dictionary of best hyperparameters found.

**Key Steps**:
1. **Parameter Loop**: For each combination of parameters in `ParameterGrid(param_grid)`, clone and fit the estimator.
2. **Retrieve Labels**: If the estimator has `.labels_`, use it; otherwise use `.predict(X)`.
3. **Compute Silhouette**: If more than one cluster is found, calculate `silhouette_score(X, labels)`.
4. **Track the Best**: Keep track of the parameter set yielding the highest silhouette score.
5. **Fallback**: If no valid parameter combos produce more than one cluster, it falls back to the original estimator.

**Example**:

```python
from utils.unsupervised_hyperparameter_tuning import clustering_hyperparameter_tuning
from sklearn.cluster import KMeans

X = ...  # Your numeric data for clustering
param_grid = {
    'model__n_clusters': [2, 3, 4],
    'model__init': ['k-means++', 'random']
}

best_model, best_params = clustering_hyperparameter_tuning(
    X, KMeans(random_state=42), param_grid, scoring='silhouette'
)
print("Best Silhouette Score found:", best_params)
```

**Note**: This approach is simpler than using `GridSearchCV` for clustering because unsupervised tasks do not have a “true” label. The silhouette score is a common measure, but you could adapt the function for other internal cluster metrics if desired.
