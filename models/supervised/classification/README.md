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

- [Logistic Regression](models/supervised/classification/logistic_regression.py)
- [Decision Tree Classifier](models/supervised/classification/decision_tree_classifier.py)
- [Random Forest Classifier (Bagging)](models/supervised/classification/random_forest_classifier.py)
- [Extra Trees Classifier](models/supervised/classification/extra_trees_classifier.py)
- [Gradient Boosting Classifier (Boosting)](models/supervised/classification/gradient_boosting_classifier.py)
- [AdaBoost Classifier (Boosting)](models/supervised/classification/adaboost_classifier.py)
- [XGBoost Classifier (Boosting)](models/supervised/classification/xgboost_classifier.py)
- [LightGBM Classifier (Boosting)](models/supervised/classification/lightgbm_classifier.py)
- [CatBoost Classifier (Boosting)](models/supervised/classification/catboost_classifier.py)
- [Support Vector Classifier (SVC)](models/supervised/classification/svc.py)
- [K-Nearest Neighbors (KNN) Classifier](models/supervised/classification/knn_classifier.py)
- [Multilayer Perceptron (MLP) Classifier](models/supervised/classification/mlp_classifier.py)
- [GaussianNB (Naive Bayes Classifier)](models/supervised/classification/gaussian_nb.py)
- [Linear Discriminant Analysis (LDA)](models/supervised/classification/linear_discriminant_analysis.py)
- [Quadratic Discriminant Analysis (QDA)](models/supervised/classification/quadratic_discriminant_analysis.py)

To train any of these models, specify the `--model_module` argument with the appropriate model name (e.g., `logistic_regression`) when running `train_classification_model.py`.
