# Gradio Interfaces

This directory contains interactive Gradio interfaces for training machine learning models, data preprocessing, and deploying trained models for predictions.

## Contents

- [`train_regressor_gradio.py`](#train_regressor_gradiopy)
- [`train_classificator_gradio.py`](#train_classificator_gradiopy)

---

### `train_regressor_gradio.py`

An interactive interface for training regression models with various algorithms and customizable configurations - https://huggingface.co/spaces/mboukabous/train_regression

#### Features

- Supports multiple regression models from the `models/supervised/regression/` directory.
- Provides three data input options:
  - Upload a dataset file.
  - Provide the path to an existing dataset.
  - Download datasets directly from Kaggle using the API.
- Allows customization of key parameters such as:
  - Test size, random state, cross-validation folds, and more.
- Automatically saves trained models, evaluation metrics, and visualizations.
- Outputs key performance metrics such as RMSE, R² score, and MAE.
- Generates an "Actual vs. Predicted" plot for regression tasks.

---

#### Usage

```bash
python train_regressor_gradio.py
```

#### Workflow

1. **Select Model Module**: Choose from available regression models.
2. **Provide Data**: Upload a CSV file, provide a dataset path, or fetch from Kaggle.
3. **Set Parameters**: Customize test size, cross-validation folds, and other options.
4. **Train Model**: Start training and monitor outputs (metrics and plots).

---

### `train_classificator_gradio.py`

An interactive interface for training classification models using various algorithms, suitable for both binary and multi-class classification tasks - https://huggingface.co/spaces/mboukabous/train_classificator

#### Features

- Supports multiple classification models from `models/supervised/classification/`.
- Flexible data input methods:
  - Upload a CSV file.
  - Provide a dataset path directly.
  - Download datasets from Kaggle using `kaggle.json`.
- Customizable parameters, including test size, random state, CV folds, and scoring metrics (e.g., `accuracy`, `f1`).
- Saves trained models and evaluation metrics.
- If `visualize` is enabled, displays classification metrics (accuracy, precision, recall, F1) and a confusion matrix plot.

---

#### Usage

```bash
python train_classificator_gradio.py
```

#### Workflow

1. **Select Model Module**: Choose from available classification models (e.g., `logistic_regression`, `random_forest_classifier`).
2. **Provide Data**: Upload, specify path, or fetch from Kaggle.
3. **Set Parameters**: Adjust test size, CV folds, scoring metric (accuracy, f1, roc_auc), etc.
4. **Train Model**: Train and review classification metrics and confusion matrix for both binary and multi-class datasets.

---

## Requirements

Install all dependencies using:
```bash
pip install -r requirements.txt
```
