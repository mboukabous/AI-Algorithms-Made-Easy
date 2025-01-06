
"""
train_anomaly_detection.py

Trains an anomaly detection model (Isolation Forest, One-Class SVM, etc.) on a dataset.
Allows dropping or selecting columns, label-encoding for non-numeric data,
saves predictions (0 = normal, 1 = outlier) and optionally visualizes in 2D.

Usage Example:
--------------
python scripts/train_anomaly_detection.py \
    --model_module isolation_forest \
    --data_path data/raw/my_dataset.csv \
    --drop_columns "unwanted_col" \
    --select_columns "feat1,feat2,feat3" \
    --visualize
"""

import os
import sys
import argparse
import importlib
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from timeit import default_timer as timer

def main(args):
    # Change to the project root if needed
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    os.chdir(project_root)
    sys.path.insert(0, project_root)

    # Dynamically import the chosen anomaly model module
    model_module_path = f"models.unsupervised.anomaly.{args.model_module}"
    model_module = importlib.import_module(model_module_path)

    # Retrieve the estimator from the model file
    estimator = model_module.estimator

    # Prepare results directory
    if args.results_path is None:
        # e.g., 'results/IsolationForest_Anomaly'
        args.results_path = os.path.join("results", f"{estimator.__class__.__name__}_Anomaly")
    os.makedirs(args.results_path, exist_ok=True)

    # Prepare model directory
    if args.model_path is None:
        # e.g., 'saved_model/IsolationForest_Anomaly'
        args.model_path = os.path.join('saved_models', f"{estimator.__class__.__name__}_Anomaly")
    os.makedirs(args.model_path, exist_ok=True)
    
    # Load data
    df = pd.read_csv(args.data_path)
    print(f"Data loaded from {args.data_path}, initial shape: {df.shape}")

    # Drop empty columns
    df = df.dropna(axis='columns', how='all')
    print("After dropping empty columns:", df.shape)

    # Drop specified columns if any
    if args.drop_columns:
        drop_cols = [c.strip() for c in args.drop_columns.split(',') if c.strip()]
        df.drop(columns=drop_cols, inplace=True, errors='ignore')
        print(f"Dropped columns: {drop_cols} | New shape: {df.shape}")

    # Select specified columns if any
    if args.select_columns:
        keep_cols = [c.strip() for c in args.select_columns.split(',') if c.strip()]
        df = df[keep_cols]
        print(f"Selected columns: {keep_cols} | New shape: {df.shape}")

    # Label-encode non-numeric columns
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Convert DataFrame to numpy array
    X = df.values
    print(f"Final data shape after dropping/selecting columns and encoding: {X.shape}")

    # Fit the anomaly model
    start_time = timer()
    estimator.fit(X)
    end_time = timer()
    train_time = end_time - start_time
    print(f"Anomaly detection training with {args.model_module} completed in {train_time:.2f} seconds.")

    # Save the model
    model_output_path = os.path.join(args.model_path, "anomaly_model.pkl")
    joblib.dump(estimator, model_output_path)
    print(f"Model saved to {model_output_path}")

    # Predict outliers: Typically returns 1 for inliers, -1 for outliers (or vice versa)
    # We'll unify them to 0 = normal, 1 = outlier
    raw_preds = estimator.predict(X)
    # Some anomaly detectors do the opposite: IsolationForest => +1 inlier, -1 outlier
    # Convert to 0/1:
    preds_binary = np.where(raw_preds == 1, 0, 1)

    outlier_count = np.sum(preds_binary)
    inlier_count = len(preds_binary) - outlier_count
    print(f"Detected {outlier_count} outliers out of {len(X)} samples. ({inlier_count} normal)")

    # Save predictions
    pred_df = pd.DataFrame({
        'OutlierPrediction': preds_binary
    })
    pred_path = os.path.join(args.results_path, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")

    # Visualization if 2D or 3D
    if args.visualize:
        print("Creating anomaly detection visualization...")
        # We'll do PCA => 2D if dimension > 2
        if X.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            x_label = "PC1"
            y_label = "PC2"
        elif X.shape[1] == 2:
            X_2d = X
            x_label = df.columns[0] if df.shape[1] == 2 else "Feature 1"
            y_label = df.columns[1] if df.shape[1] == 2 else "Feature 2"
        else:
            # 1D or 0D => skip
            print("Only 1 feature or none; can't create 2D scatter. Skipping.")
            return

        # Plot
        plt.figure(figsize=(6,5))
        # color outliers differently
        colors = np.where(preds_binary == 1, 'r', 'b')
        plt.scatter(X_2d[:,0], X_2d[:,1], c=colors, s=30, alpha=0.7)
        plt.title(f"{estimator.__class__.__name__} Anomaly Detection")
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # Save
        plot_path = os.path.join(args.results_path, "anomaly_plot.png")
        plt.savefig(plot_path)
        plt.show()
        print(f"Anomaly plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an anomaly detection model.")
    parser.add_argument('--model_module', type=str, required=True,
                        help='Name of the anomaly detection model (e.g. isolation_forest, one_class_svm).')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the CSV dataset file.')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to save the trained model.')
    parser.add_argument('--results_path', type=str, default=None,
                        help='Directory to save results (predictions, plots).')
    parser.add_argument('--drop_columns', type=str, default='',
                        help='Comma-separated column names to drop.')
    parser.add_argument('--select_columns', type=str, default='',
                        help='Comma-separated column names to keep (ignore the rest).')
    parser.add_argument('--visualize', action='store_true',
                        help='If set, reduce to 2D (via PCA if needed) and color outliers vs. normal points.')
    args = parser.parse_args()
    main(args)
