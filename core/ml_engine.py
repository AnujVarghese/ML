"""
Machine Learning Core Model Execution Engine (Compute Layer)
-----------------------------------------------------------
This module defines the core execution functions for running model benchmarks. 
It fits exactly 6 models for Regression and 6 models for Classification.

Design Rationale:
- Strict decoupling: This module operates strictly on NumPy arrays / Pandas DataFrames
  and returns pure Python and Pandas structures. Absolutely no GUI or web framework
  bindings (such as streamlit) are allowed here.
- Fault tolerance: Each model training task is wrapped in a try-except block. If a single
  estimator fails to fit (e.g. due to compatibility or memory limits), the pipeline continues,
  logs the error, and returns NaNs for that model, preventing the entire suite from crashing.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    RandomForestClassifier, 
    GradientBoostingClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    r2_score, 
    mean_absolute_error, 
    mean_squared_error,
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)

def run_regression_benchmark(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, random_state: int = 42):
    """
    Fits and evaluates exactly 6 diverse regression estimators.
    
    Args:
        X_train (pd.DataFrame): Training feature matrix.
        X_test (pd.DataFrame): Testing feature matrix.
        y_train (pd.Series): Training targets.
        y_test (pd.Series): Testing targets.
        random_state (int): Seed used by random state variables.
        
    Returns:
        tuple: (results_df, trained_models)
            - results_df: pd.DataFrame compiling R2, MAE, RMSE, and MSE for each model.
            - trained_models: dict containing fitted estimator objects.
    """
    # Define models dictionary
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=50, random_state=random_state),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=50, random_state=random_state),
        "Support Vector Regressor (SVR)": SVR(),
        "K-Neighbors Regressor": KNeighborsRegressor()
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        try:
            # Fit model on preprocessed training data
            model.fit(X_train, y_train)
            
            # Predict on testing set
            y_pred = model.predict(X_test)
            
            # Calculate standard metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            results.append({
                "Model": name,
                "R-squared (R2)": r2,
                "Mean Absolute Error (MAE)": mae,
                "Root Mean Squared Error (RMSE)": rmse,
                "Mean Squared Error (MSE)": mse
            })
            # Save the trained model for downstream real-time prediction
            trained_models[name] = model
            
        except Exception as e:
            # Gracefully handle single model failures
            results.append({
                "Model": name,
                "R-squared (R2)": np.nan,
                "Mean Absolute Error (MAE)": np.nan,
                "Root Mean Squared Error (RMSE)": np.nan,
                "Mean Squared Error (MSE)": np.nan
            })
            print(f"[ERROR] Failed to execute {name}: {e}")
            
    # Set model name as index for easy formatting in tabular displays
    results_df = pd.DataFrame(results).set_index("Model")
    return results_df, trained_models

def run_classification_benchmark(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, random_state: int = 42):
    """
    Fits and evaluates exactly 6 diverse classification estimators.
    
    Args:
        X_train (pd.DataFrame): Training feature matrix.
        X_test (pd.DataFrame): Testing feature matrix.
        y_train (pd.Series): Training targets.
        y_test (pd.Series): Testing targets.
        random_state (int): Seed used by random state variables.
        
    Returns:
        tuple: (results_df, trained_models)
            - results_df: pd.DataFrame compiling Accuracy, Precision, Recall, and F1-Score for each model.
            - trained_models: dict containing fitted estimator objects.
    """
    # Define models dictionary
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=50, random_state=random_state),
        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=50, random_state=random_state),
        "Support Vector Classifier (SVC)": SVC(probability=True, random_state=random_state),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=5, random_state=random_state)
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        try:
            # Fit model on preprocessed training data
            model.fit(X_train, y_train)
            
            # Predict classes on testing set
            y_pred = model.predict(X_test)
            
            # Calculate standard metrics (using macro average for multiclass compatibility)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            results.append({
                "Model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1
            })
            # Save the trained model for downstream real-time prediction
            trained_models[name] = model
            
        except Exception as e:
            # Gracefully handle single model failures
            results.append({
                "Model": name,
                "Accuracy": np.nan,
                "Precision": np.nan,
                "Recall": np.nan,
                "F1-Score": np.nan
            })
            print(f"[ERROR] Failed to execute {name}: {e}")
            
    # Set model name as index for easy formatting in tabular displays
    results_df = pd.DataFrame(results).set_index("Model")
    return results_df, trained_models
