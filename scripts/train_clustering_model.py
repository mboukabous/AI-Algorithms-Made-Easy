
"""
train_clustering_model.py

A script to train clustering models (K-Means, DBSCAN, Gaussian Mixture, etc.).
It can optionally perform hyperparameter tuning using silhouette score,
trains the model, saves it, and visualizes clusters if requested.
"""

import os
import sys
import argparse
import importlib
import pandas as pd
import numpy as np
import joblib

from sklearn import datasets
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from timeit import default_timer as timer

def main(args):
    # Change to the project root if needed
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(project_root)
    sys.path.insert(0, project_root)

    # Optional: import the unsupervised hyperparameter tuning function
    from utils.unsupervised_hyperparameter_tuning import clustering_hyperparameter_tuning

    # Dynamically import the chosen clustering model module
    model_module_path = f"models.unsupervised.clustering.{args.model_module}"
    model_module = importlib.import_module(model_module_path)

    # Retrieve the estimator and param grid from the model file
    estimator = model_module.estimator
    param_grid = getattr(model_module, 'param_grid', {})
    default_scoring = getattr(model_module, 'default_scoring', 'silhouette')  # fallback

    # Prepare results directory
    if args.results_path is None:
        # e.g., 'results/KMeans_Clustering'
        args.results_path = os.path.join('results', f"{estimator.__class__.__name__}_Clustering")
    os.makedirs(args.results_path, exist_ok=True)
    
    # Prepare model directory
    if args.model_path is None:
        # e.g., 'saved_model/KMeans_Clustering'
        args.model_path = os.path.join('saved_models', f"{estimator.__class__.__name__}_Clustering")
    os.makedirs(args.model_path, exist_ok=True)

    # Load data from CSV
    df = pd.read_csv(args.data_path)
    print(f"Data loaded from {args.data_path}, initial shape: {df.shape}")

    # Drop empty columns
    df = df.dropna(axis='columns', how='all')
    print("After dropping empty columns:", df.shape)

    # Drop specified columns if any
    if args.drop_columns:
        drop_cols = [col.strip() for col in args.drop_columns.split(',') if col.strip()]
        df = df.drop(columns=drop_cols, errors='ignore')
        print(f"Dropped columns: {drop_cols} | New shape: {df.shape}")

    # Select specified columns if any
    if args.select_columns:
        keep_cols = [col.strip() for col in args.select_columns.split(',') if col.strip()]
        # Keep only these columns (intersection with what's in df)
        df = df[keep_cols]
        print(f"Selected columns: {keep_cols} | New shape: {df.shape}")

    # For each non-numeric column, apply label encoding
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Convert DataFrame to NumPy array for clustering
    X = df.values
    print(f"Final shape after dropping/selecting columns and encoding: {X.shape}")

    # If user wants hyperparam tuning
    if args.tune:
        print("Performing hyperparameter tuning...")
        best_model, best_params = clustering_hyperparameter_tuning(
            X, estimator, param_grid, scoring=default_scoring, cv=args.cv_folds
        )
        estimator = best_model  # the fitted best model
        print("Best Params:", best_params)
    else:
        # Just fit the model directly
        print("No hyperparameter tuning; fitting model with default parameters...")
        start_time = timer()
        estimator.fit(X)
        end_time = timer()
        print(f"Training time (no tuning): {end_time - start_time:.2f}s")

    # Ensure the model is fitted at this point
    model_output_path = os.path.join(args.model_path, "best_model.pkl")
    joblib.dump(estimator, model_output_path)
    print(f"Model saved to {model_output_path}")

    # Evaluate using silhouette if possible
    # Some clusterers use .labels_, others require .predict(X)
    if hasattr(estimator, 'labels_'):
        labels = estimator.labels_
    else:
        labels = estimator.predict(X)  # e.g. KMeans, GaussianMixture

    unique_labels = set(labels)
    if len(unique_labels) > 1:
        sil = silhouette_score(X, labels)
        print(f"Silhouette Score: {sil:.4f}")
        pd.DataFrame({"Silhouette": [sil]}).to_csv(
            os.path.join(args.results_path, "metrics.csv"), index=False
        )
    else:
        print("Only one cluster found; silhouette score not meaningful.")

    # Visualization
    if args.visualize:
        print("Creating cluster visualization...")

        # If X has more than 2 dims, do PCA => 2D
        if X.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            var_ratio = pca.explained_variance_ratio_
            pc1_var = var_ratio[0] * 100
            pc2_var = var_ratio[1] * 100
            x_label = f"PC1 ({pc1_var:.2f}% var)"
            y_label = f"PC2 ({pc2_var:.2f}% var)"
        elif X.shape[1] == 2:
            # If we know 'df' and shape matches, label with col names
            if df.shape[1] == 2:
                x_label = df.columns[0]
                y_label = df.columns[1]
            else:
                x_label = "Feature 1"
                y_label = "Feature 2"
            X_2d = X
        else:
            # 1D or 0D => skip
            if X.shape[1] == 1:
                print("Only 1 feature available; cannot create a 2D scatter plot.")
            else:
                print("No features available for plotting.")
            return

        plt.figure(figsize=(6, 5))
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', s=30)
        plt.title(f"{estimator.__class__.__name__} Clusters")
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # Save the figure
        plot_path = os.path.join(args.results_path, "clusters.png")
        plt.savefig(plot_path)
        plt.show()
        print(f"Cluster plot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a clustering model.")
    parser.add_argument('--model_module', type=str, required=True,
                        help='Name of the clustering model module (e.g. kmeans, dbscan, etc.).')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the CSV dataset.')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to save the trained model.')
    parser.add_argument('--results_path', type=str, default=None,
                        help='Directory to save results (metrics, plots).')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='Number of folds for hyperparam tuning.')
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning with silhouette score.')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate a 2D visualization of the clusters.')
    parser.add_argument('--drop_columns', type=str, default='',
                        help='Comma-separated column names to drop from the dataset.')
    parser.add_argument('--select_columns', type=str, default='',
                        help='Comma-separated column names to keep (ignore all others).')
    args = parser.parse_args()
    main(args)
