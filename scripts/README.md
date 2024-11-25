# Scripts

This directory contains executable scripts for training, testing, and other tasks related to model development and evaluation.

## Contents

- [`train_supervised_model.py`](#train_supervised_model.py)

### `train_supervised_model.py`

A script for training supervised learning models (both regression and classification) using scikit-learn. It handles data loading, preprocessing, optional log transformation, hyperparameter tuning, model evaluation, and saving of models, metrics, and visualizations.

#### Features

- Supports various models defined in the `models/supervised` directory.
- Performs hyperparameter tuning using grid search cross-validation.
- Handles both regression and classification tasks.
- Saves trained models and evaluation metrics.
- Generates visualizations if specified.

#### Usage

```bash
python train_model.py --model_module MODEL_MODULE \
    --data_path DATA_PATH/DATA_NAME.csv \
    --target_variable TARGET_VARIABLE [OPTIONS]

```

- **Required Arguments:**
- `model_module`: Name of the model module to import (e.g., `linear_regression`).
- `data_path`: Path to the dataset directory, including the data file name.
- `target_variable`: Name of the target variable.

- **Optional Arguments:**
- `test_size`: Proportion of the dataset to include in the test split (default: 0.2).
- `random_state`: Random seed for reproducibility (default: 42).
- `log_transform`: Apply log transformation to the target variable (regression only).
- `cv_folds`: Number of cross-validation folds (default: 5).
- `scoring_metric`: Scoring metric for model evaluation.
- `model_path`: Path to save the trained model.
- `results_path`: Path to save results and metrics.
- `visualize`: Generate and save visualizations.
- `drop_columns`: Comma-separated column names to drop from the dataset.

#### Usage Example

```bash
python train_model.py --model_module random_forest_regressor \
    --data_path data/house_prices/train.csv \
    --target_variable SalePrice --drop_columns Id \
    --log_transform --visualize
```
