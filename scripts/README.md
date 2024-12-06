# Scripts

This directory contains executable scripts for training, testing, and other tasks related to model development and evaluation.

## Contents

- [`train_regression_model.py`](#train_regression_model.py)
- [`train_classification_model.py`](#train_classification_modelpy)

### `train_regression_model.py`

A script for training supervised learning **regression** models using scikit-learn. It handles data loading, preprocessing, optional log transformation, hyperparameter tuning, model evaluation, and saving of models, metrics, and visualizations.

#### Features

- Supports various regression models defined in `models/supervised/regression`.
- Performs hyperparameter tuning using grid search cross-validation.
- Saves trained models and evaluation metrics.
- Generates visualizations if specified.

#### Usage

```bash
python train_regression_model.py --model_module MODEL_MODULE \
    --data_path DATA_PATH/DATA_NAME.csv \
    --target_variable TARGET_VARIABLE [OPTIONS]

```

- **Required Arguments:**
- `model_module`: Name of the regression model module to import (e.g., `linear_regression`).
- `data_path`: Path to the dataset directory, including the data file name.
- `target_variable`: Name of the target variable.

- **Optional Arguments:**
- `test_size`: Proportion of the dataset to include in the test split (default: `0.2`).
- `random_state`: Random seed for reproducibility (default: `42`).
- `log_transform`: Apply log transformation to the target variable (regression only).
- `cv_folds`: Number of cross-validation folds (default: `5`).
- `scoring_metric`: Scoring metric for model evaluation.
- `model_path`: Path to save the trained model.
- `results_path`: Path to save results and metrics.
- `visualize`: Generate and save visualizations.
- `drop_columns`: Comma-separated column names to drop from the dataset.

#### Usage Example

```bash
python train_regression_model.py --model_module linear_regression \
    --data_path data/house_prices/train.csv \
    --target_variable SalePrice --drop_columns Id \
    --log_transform --visualize
```

---

### `train_classification_model.py`

A script for training supervised learning **classification** models using scikit-learn. It handles data loading, preprocessing, hyperparameter tuning (via grid search CV), model evaluation using classification metrics, and saving of models, metrics, and visualizations.

#### Features

- Supports various classification models defined in `models/supervised/classification`.
- Performs hyperparameter tuning using grid search cross-validation (via `classification_hyperparameter_tuning`).
- Saves trained models and evaluation metrics (accuracy, precision, recall, F1).
- If `visualize` is enabled, it generates a metrics bar chart and a confusion matrix plot.

#### Usage

```bash
python train_classification_model.py --model_module MODEL_MODULE \
    --data_path DATA_PATH/DATA_NAME.csv \
    --target_variable TARGET_VARIABLE [OPTIONS]

```

- **Required Arguments:**
- `model_module`: Name of the classification model module to import (e.g., `logistic_regression`).
- `data_path`: Path to the dataset directory, including the data file name.
- `target_variable`: Name of the target variable (categorical).

- **Optional Arguments:**
- `test_size`: Proportion of the dataset to include in the test split (default: `0.2`).
- `random_state`: Random seed for reproducibility (default: `42`).
- `cv_folds`: Number of cross-validation folds (default: `5`).
- `scoring_metric`: Scoring metric for model evaluation (e.g., `accuracy`, `f1`, `roc_auc`).
- `model_path`: Path to save the trained model.
- `results_path`: Path to save results and metrics.
- `visualize`: Generate and save visualizations.
- `drop_columns`: Comma-separated column names to drop from the dataset.

#### Usage Example

```bash
python train_classification_model.py --model_module logistic_regression \
    --data_path data/adult_income/train.csv \
    --target_variable income_bracket --drop_columns Id \
    --scoring_metric accuracy --visualize
```
