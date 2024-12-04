# Gradio Interfaces

This directory contains interactive Gradio interfaces for training machine learning models, data preprocessing, and deploying trained models for predictions.

## Contents

- [`train_regressor_gradio.py`](#train_regressor_gradio.py)

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

## Requirements

Install all dependencies using:
```bash
pip install -r requirements.txt
```
