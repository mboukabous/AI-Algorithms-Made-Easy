"""
This module provides a function for hyperparameter tuning with preprocessing using scikit-learn's GridSearchCV specifically for regression models.

Features:
- Handles numerical and categorical preprocessing using pipelines.
- Automates hyperparameter tuning for any scikit-learn regressor.
- Uses GridSearchCV for cross-validation and hyperparameter search.
- Applies algorithm-specific preprocessing when necessary.

Functions:
    - hyperparameter_tuning_model: Performs hyperparameter tuning on a given dataset and estimator.

Example Usage:
    from sklearn.ensemble import RandomForestRegressor
    from supervised_hyperparameter_tuning import hyperparameter_tuning_model

    X = ...  # Your feature DataFrame
    y = ...  # Your target variable
    param_grid = {
        'model__n_estimators': [100, 200, 500],
        'model__max_depth': [None, 10, 20]
    }
    best_model, best_params = hyperparameter_tuning_model(X, y, RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, KFold

def regression_hyperparameter_tuning(X, y, estimator, param_grid, cv=5, scoring=None):
    """
    Performs hyperparameter tuning for a given regression model using GridSearchCV with preprocessing.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        estimator: The scikit-learn regressor to use (e.g., LinearRegression(), RandomForestRegressor()).
        param_grid (dict): Hyperparameter grid for GridSearchCV.
        cv (int or cross-validation generator): Number of cross-validation folds or a cross-validation generator.
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
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Conditional preprocessing for categorical data
    estimator_name = estimator.__class__.__name__

    if estimator_name in [
        'DecisionTreeRegressor', 'RandomForestRegressor', 'ExtraTreesRegressor',
        'GradientBoostingRegressor', 'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor'
    ]:
        # Use Ordinal Encoding for tree-based models
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
    else:
        # Use OneHotEncoder for other models
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Create a pipeline that combines preprocessing and the estimator
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', estimator)
    ])

    # Define cross-validation strategy
    if isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=True, random_state=42)

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

    print(f"Best Hyperparameters for {estimator_name}:")
    for param_name in sorted(best_params.keys()):
        print(f"{param_name}: {best_params[param_name]}")

    return best_model, best_params
