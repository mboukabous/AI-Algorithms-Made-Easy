
"""
train_dimred_model.py

Trains a dimensionality reduction model (e.g., PCA, t-SNE, UMAP) on a dataset.
It can drop or select specific columns, perform label encoding on any non-numeric columns,
and optionally visualize the reduced data (2D or 3D).

Example Usage:
--------------
python scripts/train_dimred_model.py \
    --model_module pca \
    --data_path data/raw/breast-cancer-wisconsin-data/data.csv \
    --drop_columns "id" \
    --select_columns "radius_mean, texture_mean, perimeter_mean, area_mean" \
    --visualize
"""

import os
import sys
import argparse
import importlib
import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def main(args):
    # Move to project root if needed
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(project_root)
    sys.path.insert(0, project_root)

    # Dynamically import the chosen model module (pca.py, tsne.py, umap.py, etc.)
    model_module_path = f"models.unsupervised.dimred.{args.model_module}"
    model_module = importlib.import_module(model_module_path)

    # Retrieve the estimator from the model file
    estimator = model_module.estimator
    default_n_components = getattr(model_module, 'default_n_components', 2)  # fallback

    # Prepare results directory
    if args.results_path is None:
        # e.g., 'results/PCA_DimRed'
        args.results_path = os.path.join('results', f"{estimator.__class__.__name__}_DimRed")
    os.makedirs(args.results_path, exist_ok=True)

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
        df = df[keep_cols]
        print(f"Selected columns: {keep_cols} | New shape: {df.shape}")

    # Label-encode non-numeric columns
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Impute
    imputer = SimpleImputer(strategy='mean')  # or 'median'
    df_array = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(df_array, columns=df.columns)
    print("After label-encoding and imputation:", df_imputed.shape)
    
    # Convert DataFrame to numpy array
    X = df_imputed.values
    print(f"Final data shape after dropping/selecting columns and encoding: {X.shape}")

    # Fit-transform the data (typical for dimensionality reduction)
    X_transformed = estimator.fit_transform(X)
    print(f"Dimensionality reduction done using {args.model_module}. Output shape: {X_transformed.shape}")

    # Save the model
    model_output_path = os.path.join(args.results_path, "dimred_model.pkl")
    os.makedirs(args.model_path, exist_ok=True)  # ensure directory
    joblib.dump(estimator, model_output_path)
    print(f"Model saved to {model_output_path}")

    # Save the transformed data
    transformed_path = os.path.join(args.results_path, "X_transformed.csv")
    pd.DataFrame(X_transformed).to_csv(transformed_path, index=False)
    print(f"Transformed data saved to {transformed_path}")

    # Visualization (only if 2D or 3D)
    if args.visualize:
        n_dims = X_transformed.shape[1]
        if n_dims == 2:
            plt.figure(figsize=(6,5))
            plt.scatter(X_transformed[:,0], X_transformed[:,1], s=30, alpha=0.7, c='blue')
            plt.title(f"{estimator.__class__.__name__} 2D Projection")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plot_path = os.path.join(args.results_path, "dimred_plot_2D.png")
            plt.savefig(plot_path)
            plt.show()
            print(f"2D plot saved to {plot_path}")
        elif n_dims == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_transformed[:,0], X_transformed[:,1], X_transformed[:,2], s=30, alpha=0.7, c='blue')
            ax.set_title(f"{estimator.__class__.__name__} 3D Projection")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
            plot_path = os.path.join(args.results_path, "dimred_plot_3D.png")
            plt.savefig(plot_path)
            plt.show()
            print(f"3D plot saved to {plot_path}")
        else:
            print(f"Visualization only supported for 2D or 3D outputs. Got {n_dims}D, skipping.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a dimensionality reduction model.")
    parser.add_argument('--model_module', type=str, required=True,
                        help='Name of the dimred model module (e.g. pca, tsne, umap).')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the CSV dataset file.')
    parser.add_argument('--model_path', type=str, default='saved_models/DimRed',
                        help='Where to save the fitted model.')
    parser.add_argument('--results_path', type=str, default=None,
                        help='Directory to store results (transformed data, plots).')
    parser.add_argument('--visualize', action='store_true',
                        help='Plot the transformed data if 2D or 3D.')
    parser.add_argument('--drop_columns', type=str, default='',
                        help='Comma-separated column names to drop from the dataset.')
    parser.add_argument('--select_columns', type=str, default='',
                        help='Comma-separated column names to keep (ignore the rest).')

    args = parser.parse_args()
    main(args)
