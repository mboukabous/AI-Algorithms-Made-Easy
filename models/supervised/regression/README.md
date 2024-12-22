# Regression Models

This directory contains Python scripts that define various regression models and their associated hyperparameter grids. Each model file sets up a scikit-learn-compatible estimator and defines a parameter grid for use with the `train_regression_model.py` script.

These model definition files:
- Specify an estimator (e.g., `LinearRegression()`, `RandomForestRegressor()`, etc.).
- Define a `param_grid` dict for hyperparameter tuning using `GridSearchCV`.
- Optionally provide a `default_scoring` metric.
- Are intended to be flexible and modular, allowing you to easily swap out models without changing other parts of the code.

**Note:** Preprocessing steps (imputation, scaling, encoding) and hyperparameter tuning logic are handled externally by the scripts and utilities.

## Available Regression Models

- [Linear Regression](linear_regression.py)
- [Ridge Regression](ridge_regression.py)
- [Lasso Regression](lasso_regression.py)
- [ElasticNet Regression](elasticnet_regression.py)
- [Decision Tree Regressor](decision_tree_regressor.py)
- [Random Forest Regressor (Bagging)](random_forest_regressor.py)
- [Gradient Boosting Regressor (Boosting)](gradient_boosting_regressor.py)
- [AdaBoost Regressor (Boosting)](adaboost_regressor.py)
- [XGBoost Regressor (Boosting)](xgboost_regressor.py)
- [LightGBM Regressor](lightgbm_regressor.py)
- [CatBoost Regressor](catboost_regressor.py)
- [Support Vector Regressor (SVR)](support_vector_regressor.py)
- [K-Nearest Neighbors (KNN) Regressor](knn_regressor.py)
- [Extra Trees Regressor](extra_trees_regressor.py)
- [Multilayer Perceptron (MLP) Regressor](mlp_regressor.py)

To train any of these models, specify the `--model_module` argument with the appropriate model name (e.g., `linear_regression`) when running `train_regression_model.py`.
