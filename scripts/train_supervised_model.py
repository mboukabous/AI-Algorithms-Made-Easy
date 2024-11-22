
"""
This script trains supervised learning models using scikit-learn.
It handles both regression and classification tasks, including
data loading, preprocessing, optional log transformation (for
regression), hyperparameter tuning, model evaluation, and saving
of models, metrics, and visualizations.

Usage:
    python train_model.py --model_module MODEL_MODULE --data_path DATA_PATH
                          --data_name DATA_NAME.csv --target_variable TARGET_VARIABLE

Optional arguments:
    --test_size TEST_SIZE
    --random_state RANDOM_STATE
    --log_transform (only for regression)
    --cv_folds CV_FOLDS
    --scoring_metric SCORING_METRIC
    --model_path MODEL_PATH
    --results_path RESULTS_PATH
    --visualize
    --drop_columns COLUMN_NAMES

Example:
    python train_model.py --model_module linear_regression
                          --data_path data/house_prices --data_name train.csv
                          --target_variable SalePrice --drop_columns Id
                          --log_transform --visualize
"""

import os
import sys
import argparse
import importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (root_mean_squared_error, mean_squared_error, r2_score, mean_absolute_error,
                             accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay)
import joblib

def main(args):
    # Change to the root directory of the project
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(project_root)
    sys.path.insert(0, project_root)

    # Import the hyperparameter tunine and the model modules
    from utils.supervised_hyperparameter_tuning import hyperparameter_tuning_model
    model_module_path = f"models.supervised.{args.model_module}"
    model_module = importlib.import_module(model_module_path)
    
    # Get the model estimator, parameters grid, and the scoring metric
    estimator = model_module.estimator
    param_grid = model_module.param_grid
    scoring_metric = args.scoring_metric or getattr(model_module, 'default_scoring', None)
    model_name = estimator.__class__.__name__

    # Set default paths if not provided
    args.model_path = args.model_path or os.path.join('models', model_name)
    args.results_path = args.results_path or os.path.join('results', model_name)
    os.makedirs(args.results_path, exist_ok=True)

    # Load the dataset
    df = pd.read_csv(os.path.join(args.data_path, args.data_name))

    # Drop specified columns
    if args.drop_columns:
        columns_to_drop = args.drop_columns.split(',')
        df = df.drop(columns=columns_to_drop)

    # Define target variable and features
    target_variable = args.target_variable
    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    # Determine if the task is regression or classification
    if y.dtype.kind in 'fi':  # If it's numeric
        if len(np.unique(y)) <= 2:  # Check the number of unique values
            task_type = 'classification' # you can put "binary classification" here
        else:
            task_type = 'regression'
    else:
        task_type = 'classification'  # Default for non-numeric types

    # Encode target variable if classification and not numeric
    if task_type == 'classification' and y.dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        # Save label encoder for inverse transformation
        joblib.dump(le, os.path.join(args.model_path, 'label_encoder.pkl'))

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state)

    # Visualize target variable distribution (regression)
    if args.visualize and task_type == 'regression':
        plt.figure(figsize=(6, 4))
        sns.histplot(y_train, kde=True)
        plt.title(f'{target_variable} Distribution Before Transformation')
        plt.savefig(os.path.join(args.results_path, 'target_distribution_before.png'))
        plt.show()

    # Optional: Apply log transformation (regression)
    if args.log_transform and task_type == 'regression':
        y_train_transformed = np.log1p(y_train)
        y_test_transformed = np.log1p(y_test)
        if args.visualize:
            plt.figure(figsize=(6, 4))
            sns.histplot(y_train_transformed, kde=True, color='green')
            plt.title(f'{target_variable} Distribution After Log Transform')
            plt.savefig(os.path.join(args.results_path, 'target_distribution_after.png'))
            plt.show()
    else:
        y_train_transformed = y_train
        y_test_transformed = y_test

    # Perform hyperparameter tuning
    best_model, best_params = hyperparameter_tuning_model(
        X_train, y_train_transformed, estimator, param_grid,
        cv=args.cv_folds, scoring=scoring_metric)

    # Evaluate the best model on the test set
    y_pred_transformed = best_model.predict(X_test)

    # Reverse transformation if applied (regression)
    if args.log_transform and task_type == 'regression':
        y_pred = np.expm1(y_pred_transformed)
        y_test_actual = np.expm1(y_test_transformed)
    else:
        y_pred = y_pred_transformed
        y_test_actual = y_test_transformed

    # Save the trained model
    model_output_path = os.path.join(args.model_path, 'best_model.pkl')
    os.makedirs(args.model_path, exist_ok=True)
    joblib.dump(best_model, model_output_path)
    print(f"Trained model saved to {model_output_path}")

    # Calculate metrics
    if task_type == 'regression':
        rmse = root_mean_squared_error(y_test_actual, y_pred)
        r2 = r2_score(y_test_actual, y_pred)
        mae = mean_absolute_error(y_test_actual, y_pred)
        mse = mean_squared_error(y_test_actual, y_pred)
        print(f"\n{model_name} Regression Metrics on Test Set:")
        print(f"- RMSE: {rmse:.4f}")
        print(f"- R² Score: {r2:.4f}")
        print(f"- MAE: {mae:.4f}")
        print(f"- MSE: {mse:.4f}")
        # Save metrics
        metrics = {'RMSE': [rmse], 'R2': [r2], 'MAE': [mae], 'MSE': [mse]}
    else:
        accuracy = accuracy_score(y_test_actual, y_pred)
        precision = precision_score(y_test_actual, y_pred, average='weighted')
        recall = recall_score(y_test_actual, y_pred, average='weighted')
        f1 = f1_score(y_test_actual, y_pred, average='weighted')
        print(f"\n{model_name} Classification Metrics on Test Set:")
        print(f"- Accuracy: {accuracy:.4f}")
        print(f"- Precision: {precision:.4f}")
        print(f"- Recall: {recall:.4f}")
        print(f"- F1 Score: {f1:.4f}")
        # Save metrics
        metrics = {'Accuracy': [accuracy], 'Precision': [precision], 'Recall': [recall], 'F1 Score': [f1]}

    # Save metrics to CSV
    results_df = pd.DataFrame(metrics)
    results_df.to_csv(os.path.join(args.results_path, 'metrics.csv'), index=False)
    print(f"\nMetrics saved to {os.path.join(args.results_path, 'metrics.csv')}")

    if args.visualize:
        if task_type == 'regression':
            # Plot Actual vs. Predicted (regression)
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test_actual, y_pred, alpha=0.6, color='blue')
            plt.plot([y_test_actual.min(), y_test_actual.max()],
                    [y_test_actual.min(), y_test_actual.max()], 'r--')
            plt.xlabel(f'Actual {target_variable}')
            plt.ylabel(f'Predicted {target_variable}')
            plt.title(f'Actual vs. Predicted {target_variable}')
            plt.savefig(os.path.join(args.results_path, 'actual_vs_predicted.png'))
            plt.show()
            print(f"Visualization saved to {os.path.join(args.results_path, 'actual_vs_predicted.png')}")
        else:
            # Plot Classification Metrics (classification)
            plt.figure(figsize=(8, 6))
            
            # Extract metrics and values
            metric_names = list(metrics.keys())
            metric_values = [value[0] for value in metrics.values()]  # Extract the single value from each list

            # Create bar chart
            plt.bar(metric_names, metric_values, color='skyblue', alpha=0.8)
            plt.ylim(0, 1)  # Metrics like accuracy, precision, etc., are typically between 0 and 1
            plt.xlabel('Metrics')
            plt.ylabel('Scores')
            plt.title('Classification Metrics')
            
            # Save and display the plot
            plt.savefig(os.path.join(args.results_path, 'classification_metrics.png'))
            plt.show()
            print(f"Visualization saved to {os.path.join(args.results_path, 'classification_metrics.png')}")

            # Display and save the confusion matrix
            conf_matrix = confusion_matrix(y_test_actual, y_pred) # Compute the confusion matrix

            disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
            disp.plot(cmap=plt.cm.Blues, values_format='d')  # Format as integers for counts
            plt.title(f'{model_name} Confusion Matrix')
            
            # Save the confusion matrix plot
            conf_matrix_path = os.path.join(args.results_path, 'confusion_matrix.png')
            plt.savefig(conf_matrix_path)
            plt.show()
            print(f"Confusion matrix saved to {conf_matrix_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a supervised model.")
    # Model module argument
    parser.add_argument('--model_module', type=str, required=True,
                        help='Name of the model module to import.')
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the dataset directory.')
    parser.add_argument('--data_name', type=str, required=True,
                        help='Name of the data file.')
    parser.add_argument('--target_variable', type=str, required=True,
                        help='Name of the target variable.')
    parser.add_argument('--drop_columns', type=str, default='',
                        help='Columns to drop from the dataset.')
    # Model arguments
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion for test split.')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--log_transform', action='store_true',
                        help='Apply log transformation (regression only).')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='Number of cross-validation folds.')
    parser.add_argument('--scoring_metric', type=str, default=None,
                        help='Scoring metric for model evaluation.')
    # Output arguments
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to save the trained model.')
    parser.add_argument('--results_path', type=str, default=None,
                        help='Path to save results and metrics.')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate and save visualizations.')

    args = parser.parse_args()
    main(args)
