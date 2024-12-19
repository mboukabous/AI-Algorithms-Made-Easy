# Machine Learning Regression Models Documentation

## 1. Introduction

Welcome to the **Machine Learning Regression Models Documentation**! This guide provides detailed instructions on how to use various regression algorithms implemented in Python using scikit-learn and other popular libraries. The scripts are designed to be flexible and reusable, allowing you to train and evaluate different models on your dataset.

**Key Features:**

- Modular design with separate scripts for each model.
- Hyperparameter tuning using `GridSearchCV`.
- Preprocessing pipelines for numerical and categorical data.
- Support for external libraries like XGBoost, LightGBM, and CatBoost.
- Visualization of results and metrics.
- Utility scripts for data download and preparation.

---

## 2. Installation and Setup

### Prerequisites

- **Python 3.7 or higher**
- **pip** package manager
- **Kaggle API credentials** (`kaggle.json` file): if you want to use datasets from Kaggle.

### Required Libraries

Install the required libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**Contents of `requirements.txt`:**

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
```

### Project Structure

Your project directory should be organized as follows:

```
project_root/
├── data/
│   ├── raw/
│   │   └── dataset/
│   │       └── data_name.csv
│   └── datasets/
│       └── kaggle_data.py
├── interfaces/
│   └── gradio/
│       └── train_regressor_gradio.py
├── models/
│   └── supervised/
│       └── regression/
│           ├── linear_regression.py
│           ├── ridge_regression.py
│           ├── lasso_regression.py
│           ├── elasticnet_regression.py
│           ├── decision_tree_regressor.py
│           ├── random_forest_regressor.py
│           ├── gradient_boosting_regressor.py
│           ├── adaboost_regressor.py
│           ├── xgboost_regressor.py
│           ├── lightgbm_regressor.py
│           ├── catboost_regressor.py
│           ├── support_vector_regressor.py
│           ├── knn_regressor.py
│           ├── extra_trees_regressor.py
│           └── mlp_regressor.py
├── scripts/
│   └── train_regression_model.py
├── utils/
│   └── supervised_hyperparameter_tuning.py
├── saved_models/
├── results/
├── requirements.txt
└── README.md
```

---

## 3. Data Preparation

### Dataset Requirements

- The dataset should be in **CSV format**.
- It should include a **target variable** for regression.
- Features can be a mix of **numerical** and **categorical** data.
- Ensure there are **no missing target values**.

### Example Dataset

We'll use the **House Prices** dataset from Kaggle as an example. You can download it using the provided script.

### Downloading the Dataset

To download the dataset using the Kaggle API, you need to have a Kaggle account and API credentials (`kaggle.json` file).

**Steps:**

1. **Obtain Kaggle API Credentials:**

   - Log in to your Kaggle account.
   - Go to "My Account" and select "Create New API Token".
   - This will download a `kaggle.json` file containing your API credentials.

2. **Use the Provided Script to Download the Dataset:**

   ```python
   # Download and Get the dataset
   from data.datasets.kaggle_data import get_kaggle_data

   JSON_KAGGLE_PATH = "/path/to/your/kaggle.json"  # Update this path
   DATA_NAME = "house-prices-advanced-regression-techniques"

   competition_path = get_kaggle_data(JSON_KAGGLE_PATH, DATA_NAME, is_competition=True)
   print(f"Dataset is available at: {competition_path}")
   ```

   **Note:** Replace `"/path/to/your/kaggle.json"` with the actual path to your `kaggle.json` file.

### Disclaimers and Notes

- **Dataset Size and Resource Usage:**

  - **Large Datasets:** Be cautious when working with large datasets, as they can consume significant memory (RAM) and processing power.
  - **Training Time:** Large datasets and complex models can lead to long training times.
  - **Resource Limitations:** If you encounter issues like RAM crashes or extremely slow training, consider:

    - **Reducing the Parameter Grid:** Limit the number of hyperparameter combinations in the `param_grid` to reduce computational load.
    - **Using a Subset of Data:** Use a smaller sample of the dataset for initial experiments.
    - **Upgrading Hardware:** If possible, use a machine with more RAM or better CPU/GPU capabilities.
    - **Cloud Computing:** Utilize cloud services like Google Colab, AWS, or Azure for more computational resources.

- **Kaggle API Limitations:**

  - Ensure that you comply with Kaggle's terms of service when downloading and using datasets.
  - Some competitions may have rules regarding data usage and sharing.

---

## 4. Model Training and Evaluation

The main script for training models is `scripts/train_regression_model.py`. It handles:

- Data loading and preprocessing.
- Optional log transformation of the target variable.
- Hyperparameter tuning using `GridSearchCV`.
- Model evaluation and metrics calculation.
- Saving trained models and results.
- Visualization of results.

### Running the Training Script

Use the following command to train a model:

```bash
python scripts/train_regression_model.py \
    --model_module 'linear_regression' \
    --data_path 'data/raw/house-prices-advanced-regression-techniques/train.csv' \
    --target_variable 'SalePrice' \
    --drop_columns 'Id' \
    --test_size 0.2 \
    --random_state 42 \
    --log_transform \
    --cv_folds 5 \
    --scoring_metric 'neg_root_mean_squared_error' \
    --model_path 'saved_models/LinearRegression' \
    --results_path 'results/LinearRegression' \
    --visualize
```

**Required Command-Line Arguments:**
- `--model_module`: Name of the model module to import (e.g., `linear_regression`).
- `--data_path`: Path to the dataset directory, including the data file name.
- `--target_variable`: Name of the target variable in your dataset.

**Optional Command-Line Arguments:**
- `--test_size`: Proportion of the dataset to include in the test split (default: `0.2`).
- `--random_state`: Random seed for reproducibility (default: `42`).
- `--log_transform`: Apply log transformation to the target variable (regression only).
- `--cv_folds`: Number of cross-validation folds (default: `5`).
- `--scoring_metric`: Scoring metric for model evaluation (e.g., `neg_root_mean_squared_error`, `accuracy`).
- `--model_path`: Path to save the trained model (e.g., `saved_models/LinearRegression`).
- `--results_path`: Path to save results and metrics (e.g., `results/LinearRegression`).
- `--visualize`: Generate and save visualizations (e.g., actual vs. predicted plots for regression).
- `--drop_columns`: Comma-separated column names to drop from the dataset (optional).

---

## 5. Usage Examples

### Simple Example: Linear Regression

```bash
python scripts/train_regression_model.py \
    --model_module 'linear_regression' \
    --data_path 'data/raw/house-prices-advanced-regression-techniques/train.csv' \
    --target_variable 'SalePrice' \
    --drop_columns 'Id' \
    --log_transform \
    --visualize
```

### Complexe Example: Linear Regression

```bash
python scripts/train_regression_model.py \
    --model_module 'linear_regression' \
    --data_path 'data/raw/house-prices-advanced-regression-techniques/train.csv' \
    --target_variable 'SalePrice' \
    --drop_columns 'Id' \
    --log_transform \
    --test_size 0.20 \
    --random_state 42 \
    --cv_folds 5 \
    --scoring_metric 'neg_mean_absolute_error' \
    --model_path 'saved_models/LinearRegression' \
    --results_path 'results/LinearRegression' \
    --visualize
```

### Simple Example: Ridge Regression

```bash
python scripts/train_regression_model.py \
    --model_module 'ridge_regression' \
    --data_path 'data/raw/house-prices-advanced-regression-techniques/train.csv' \
    --target_variable 'SalePrice' \
    --drop_columns 'Id' \
    --log_transform \
    --visualize
```

**Note:** For models that require external libraries (e.g., XGBoost, LightGBM, CatBoost), ensure the libraries are installed.

---

## 6. Training Models Using the Gradio Interface

### Introduction

The Gradio interface provides an easy and interactive way to train regression models without writing code or using the command line. It allows you to select models, configure parameters, upload data, and view results—all through a user-friendly web interface - https://huggingface.co/spaces/mboukabous/train_regression

![Interface](/interfaces/gradio//img/train_regressor_gradio.png?raw=true "Interface")

**Steps to Use the Gradio Interface**:

### Launch the Interface
Run the following command in your terminal:
```bash
python interfaces/gradio/train_regressor_gradio.py
```
This will start the Gradio app and provide a local and global URLs. Open one of those URLs in your web browser to access the interface.

### Select a Model and Configure Parameters
- **Select Model Module**: Choose the regression model you want to train from the dropdown menu (e.g., `linear_regression`, `random_forest_regressor`).
- **Set Parameters**:
   - **Scoring Metric**: Specify the metric for evaluating the model (default is `neg_root_mean_squared_error`).
   - **Test Size**: Adjust the proportion of data used for testing.
   - **Random State**: Set a seed for reproducibility.
   - **CV Folds**: Choose the number of cross-validation folds.
   - **Log Transform Target Variable**: Check this if you want to apply a log transformation to the target variable.
   - **Generate Visualizations**: Enable this to create plots after training.

### Provide Data Input
- **Upload Data File**:
   - Click "Upload CSV Data File" to upload your dataset.
- **Download from Kaggle (if applicable)**:
   - Upload your kaggle.json file.
   - Enter the Kaggle competition name and data file name (e.g., train.csv).

### Update Columns and Select Features
- Click "**Update Columns**" to load the dataset's column names.
- **Select Target Variable**: Choose the column that you want to predict.
- **Columns to Drop**: Select any columns you want to exclude from the training.

### Train the Model
- Click "Train Model" to start the training process.
- The output section will display training logs and results.
- If visualizations are enabled, plots like the actual vs. predicted values will be shown.

---

## 7. Common Issues and Troubleshooting

### Convergence Warnings

- **Cause:** Some models (e.g., Lasso, ElasticNet, MLP) may not converge within the default number of iterations.
- **Solution:** Increase `max_iter` in the hyperparameter grid.

### Solver Compatibility Issues

- **Cause:** Certain solvers (e.g., 'svd', 'cholesky' in Ridge Regression) may not support sparse data.
- **Solution:** Set `sparse_output=False` in `OneHotEncoder` to output dense arrays.

### Feature Scaling

- **Important for:** SVR, KNN, MLP.
- **Solution:** Ensure `StandardScaler` is used in the preprocessing pipeline.

### External Libraries Not Installed

- **Cause:** Models like XGBoost, LightGBM, CatBoost require external libraries.
- **Solution:** Install the required libraries using `pip`.

  ```bash
  pip install xgboost lightgbm catboost
  ```

### Handling Categorical Features

- **CatBoost:** Handles categorical features natively.
- **Solution:** Adjust preprocessing pipeline to exclude one-hot encoding when using CatBoost.

### Resource Limitations

- **Issue:** RAM crashes or slow training due to large datasets or complex models.
- **Solution:**

  - Reduce the number of hyperparameter combinations.
  - Use simpler models for initial experiments.
  - Upgrade hardware or use cloud computing resources.

---

## 8. Conclusion

This documentation provides a comprehensive guide to using various regression algorithms with your machine learning scripts. By following the instructions and understanding the special considerations for each model, you can effectively train and evaluate models on your dataset.

Feel free to experiment with different models and hyperparameters to find the best fit for your data.

---

## Additional Resources

- **Scikit-learn Documentation:** [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)
- **XGBoost Documentation:** [https://xgboost.readthedocs.io/en/latest/](https://xgboost.readthedocs.io/en/latest/)
- **LightGBM Documentation:** [https://lightgbm.readthedocs.io/en/latest/](https://lightgbm.readthedocs.io/en/latest/)
- **CatBoost Documentation:** [https://catboost.ai/docs/](https://catboost.ai/docs/)
- **Kaggle API Documentation:** [https://www.kaggle.com/docs/api](https://www.kaggle.com/docs/api)
- **Gradio Documentation:** [https://gradio.app/docs/](https://gradio.app/docs/)

---

## Contact

If you have any questions or need further assistance, please feel free to reach out.

---
