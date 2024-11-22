
"""
This module provides a function for hyperparameter tuning with preprocessing using scikit-learn's GridSearchCV.

Features:
- Handles numerical and categorical preprocessing using pipelines.
- Automates hyperparameter tuning for any scikit-learn estimator.
- Uses GridSearchCV for cross-validation and hyperparameter search.

Functions:
    - hyperparameter_tuning_model: Performs hyperparameter tuning on a given dataset and estimator.

Example Usage:
    from sklearn.ensemble import RandomForestRegressor
    from supervised_hyperparameter_tuning import hyperparameter_tuning_model

    X = ...  # Your feature DataFrame
    y = ...  # Your target variable
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20]
    }
    best_model, best_params = hyperparameter_tuning_model(X, y, RandomForestRegressor(), param_grid, preprocessor, cv=5, scoring='neg_mean_squared_error')
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV

def hyperparameter_tuning_model(X, y, estimator, param_grid, preprocessor, cv=5, scoring=None):
    """
    Performs hyperparameter tuning for a given model using GridSearchCV with preprocessing.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        estimator: The scikit-learn estimator to use (e.g., LinearRegression(), RandomForestRegressor()).
        param_grid (dict): Hyperparameter grid for GridSearchCV.
        cv (int): Number of cross-validation folds.
        scoring (str or None): Scoring metric to use.

    Returns:
        best_model (Pipeline): Best model within a pipeline from GridSearch.
        best_params (dict): Best hyperparameters.
    """
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Define preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler())
    ])

    # Define preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a pipeline that combines preprocessing and the estimator
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', estimator)
    ])

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    # Perform Grid Search
    grid_search.fit(X, y)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"Best Hyperparameters for {estimator.__class__.__name__}:")
    for param_name in sorted(best_params.keys()):
        print(f"{param_name}: {best_params[param_name]}")

    return best_model, best_params
