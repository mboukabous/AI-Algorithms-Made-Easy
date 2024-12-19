# Machine Learning Classification Models Documentation

## 1. Introduction

Welcome to the **Machine Learning Classification Models Documentation**! This guide provides step-by-step instructions on how to use various classification algorithms implemented in Python using scikit-learn and other popular libraries. The scripts are designed to be flexible and reusable, allowing you to train and evaluate different models on your dataset for both binary and multi-class classification tasks.

**Key Features:**

- Modular design with separate scripts for each classification model.
- Hyperparameter tuning using `GridSearchCV`.
- Preprocessing pipelines for numerical and categorical data.
- Support for well-known libraries (XGBoost, LightGBM, CatBoost) in addition to scikit-learn built-ins.
- Visualization of classification results and metrics (e.g., confusion matrix).
- Utility scripts for data download and preparation.
- Gradio interface for interactive model training and evaluation.

---

## 2. Installation and Setup

### Prerequisites

- **Python 3.7 or higher**
- **pip** package manager
- **Kaggle API credentials** (`kaggle.json` file) if you want to download datasets from Kaggle.

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
│       └── train_classificator_gradio.py
├── models/
│   └── supervised/
│       └── classification/
│           ├── logistic_regression.py
│           ├── decision_tree_classifier.py
│           ├── random_forest_classifier.py
│           ├── extra_trees_classifier.py
│           ├── gradient_boosting_classifier.py
│           ├── adaboost_classifier.py
│           ├── xgboost_classifier.py
│           ├── lightgbm_classifier.py
│           ├── catboost_classifier.py
│           ├── svc.py
│           ├── knn_classifier.py
│           ├── mlp_classifier.py
│           ├── gaussian_nb.py
│           ├── linear_discriminant_analysis.py
│           └── quadratic_discriminant_analysis.py
├── scripts/
│   └── train_classification_model.py
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
- It should include a **target variable** for classification.
- The target variable can be **binary** (two classes) or **multi-class** (more than two classes).
- Features can be a mix of **numerical** and **categorical** data.
- Ensure there are **no missing target values**.

### Example Datasets

- **Binary Classification**: The **Adult Income** dataset (predicting whether income > 50K) is a real-world scenario that includes both numerical and categorical features.
- **Multi-Class Classification**: The **Otto Group Product Classification** dataset is a popular multi-class problem, allowing you to demonstrate multi-class metrics and performance.

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
   DATA_NAME = "otto-group-product-classification-challenge"

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

The main script for training models is `scripts/train_classification_model.py`. It handles:

- Data loading and preprocessing.
- Hyperparameter tuning using `GridSearchCV`.
- Model evaluation using classification metrics (Accuracy, Precision, Recall, F1-score).
- Saving trained models and results.
- Visualization of metrics and confusion matrices if specified.

### Running the Training Script

Use the following command to train a model:

```bash
python scripts/train_classification_model.py \
    --model_module 'logistic_regression' \
    --data_path 'data/raw/adult-income-dataset/adult.csv' \
    --target_variable 'income' \
    --test_size 0.2 \
    --random_state 42 \
    --cv_folds 5 \
    --scoring_metric 'accuracy' \
    --model_path 'saved_models/LogisticRegression' \
    --results_path 'results/LogisticRegression' \
    --visualize
```

**Required Command-Line Arguments:**
- `--model_module`: Name of the classification model module to import (e.g., `logistic_regression`).
- `--data_path`: Path to the dataset directory, including the data file name.
- `--target_variable`: Name of the target variable in your dataset (categorical).

**Optional Command-Line Arguments:**
- `--test_size`: Proportion of the dataset to include in the test split (default: `0.2`).
- `--random_state`: Random seed for reproducibility (default: `42`).
- `--cv_folds`: Number of cross-validation folds (default: `5`).
- `--scoring_metric`: Metric for evaluation (e.g., `accuracy`, `f1`, `f1_macro`, `roc_auc`).
- `--model_path`: Path to save the trained model (e.g., `saved_models/LogisticRegression`).
- `--results_path`: Path to save results and metrics (e.g., `results/LogisticRegression`).
- `--visualize`: Generate and save classification metrics chart and confusion matrix.
- `--drop_columns`: Comma-separated column names to drop from the dataset (optional).

### Multi-Class vs. Binary Classification
The same script and pipeline handle both binary and multi-class classification. Simply choose a dataset with the appropriate target variable and, if desired, specify a multi-class-friendly metric like `f1_macro`.

---

## 5. Usage Examples

### Binary Classification Example: Logistic Regression

```bash
python scripts/train_classification_model.py \
    --model_module 'logistic_regression' \
    --data_path 'data/raw/adult-income-dataset/adult.csv' \
    --target_variable 'income' \
    --visualize
```

### Multi-Class Classification Example: Random Forest

```bash
python scripts/train_classification_model.py \
    --model_module 'random_forest_classifier' \
    --data_path 'data/raw/otto-group-product-classification-challenge/train.csv' \
    --target_variable 'target' \
    --drop_columns 'id' \
    --visualize
```

**Note:** For models that require external libraries (e.g., XGBoost, LightGBM, CatBoost), ensure the libraries are installed.

---

## 6. Training Models Using the Gradio Interface

### Introduction

The Gradio interface provides an interactive way to train classification models without using the command line. It allows for easy model selection, parameter tuning, and data handling.

https://huggingface.co/spaces/mboukabous/train_classificator

![Interface](/interfaces/gradio/img/train_classificator_gradio.png?raw=true "Interface")

**Steps to Use the Gradio Interface**:

### Launch the Interface
Run the following command in your terminal:
```bash
python interfaces/gradio/train_classificator_gradio.py
```
This will start the Gradio app and provide a local and global URLs. Open one of those URLs in your web browser to access the interface.

### Select a Model and Configure Parameters
- **Select Model Module**: Choose a classification model from the dropdown (e.g., `logistic_regression`, `random_forest_classifier`).
- **Set Parameters**:
   - **Scoring Metric**: Specify the metric for evaluating the model (default is `accuracy`).
   - **Test Size**: Adjust the proportion of data used for testing.
   - **Random State**: Set a seed for reproducibility.
   - **CV Folds**: Choose the number of cross-validation folds.
   - **Generate Visualizations**: Enable this to get metrics charts and confusion matrices.

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
- If visualizations are enabled, a confusion matrix and a classification metrics bar chart are shown.

---

## 7. Common Issues and Troubleshooting

### Convergence Warnings

- **Cause:** Some models may not converge within the default number of iterations.
- **Solution:** Increase `max_iter` in the hyperparameter grid.

### Metrics and Multi-class Tasks

- **Cause:** Multi-class classification requires metrics that account for multiple classes (e.g., `f1_macro` or `f1_weighted`).
- **Solution:** Use appropriate metrics like `f1_macro` for balanced datasets or `f1_weighted` for imbalanced datasets.
- **Note:** Ensure the target variable has more than two unique classes when performing multi-class classification to avoid misconfiguration.

### External Libraries Not Installed

- **Cause:** Models like XGBoost, LightGBM, CatBoost require external libraries.
- **Solution:** Install the required libraries using `pip`.

  ```bash
  pip install xgboost lightgbm catboost
  ```

### Resource Limitations

- **Issue:** RAM crashes or slow training due to large datasets or complex models.
- **Solution:**

  - Reduce the number of hyperparameter combinations.
  - Use simpler models for initial experiments.
  - Upgrade hardware or use cloud computing resources.

---

## 8. Conclusion

This documentation guides you through using a flexible, modular system for training classification models. By following these steps, you can easily experiment with different models, hyperparameters, and datasets—both binary and multi-class.

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
