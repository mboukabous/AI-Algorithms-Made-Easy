# Classification Models

This directory contains Python scripts that define various classification models and their associated hyperparameter grids. Each model file sets up a scikit-learn-compatible estimator and defines a parameter grid for use with the `train_classification_model.py` script.

These model definition files:
- Specify an estimator (e.g., `LogisticRegression()`, `RandomForestClassifier()`, `XGBClassifier()`).
- Define a `param_grid` dict for hyperparameter tuning using `GridSearchCV`.
- Optionally provide a `default_scoring` metric (e.g., `accuracy`).
- Work for both binary and multi-class classification tasks.
- Are intended to be flexible and modular, allowing easy swapping of models without changing other parts of the code.

**Note:** Preprocessing steps, hyperparameter tuning logic, and label encoding for categorical targets are handled externally by the scripts and utilities.

## Available Classification Models

- [Logistic Regression](logistic_regression.py)
- [Decision Tree Classifier](decision_tree_classifier.py)
- [Random Forest Classifier (Bagging)](random_forest_classifier.py)
- [Extra Trees Classifier](extra_trees_classifier.py)
- [Gradient Boosting Classifier (Boosting)](gradient_boosting_classifier.py)
- [AdaBoost Classifier (Boosting)](adaboost_classifier.py)
- [XGBoost Classifier (Boosting)](xgboost_classifier.py)
- [LightGBM Classifier (Boosting)](lightgbm_classifier.py)
- [CatBoost Classifier (Boosting)](catboost_classifier.py)
- [Support Vector Classifier (SVC)](svc.py)
- [K-Nearest Neighbors (KNN) Classifier](knn_classifier.py)
- [Multilayer Perceptron (MLP) Classifier](mlp_classifier.py)
- [GaussianNB (Naive Bayes Classifier)](gaussian_nb.py)
- [Linear Discriminant Analysis (LDA)](linear_discriminant_analysis.py)
- [Quadratic Discriminant Analysis (QDA)](quadratic_discriminant_analysis.py)

To train any of these models, specify the `--model_module` argument with the appropriate model name (e.g., `logistic_regression`) when running `train_classification_model.py`.
