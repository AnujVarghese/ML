"""
Data Preprocessing Utility Engine (Transformation Layer)
---------------------------------------------------------
This module cleans, imputes, encodes, and scales dataset columns. 

Design Rationale:
- Preventing Data Leakage: We perform the train/test split FIRST. Any imputers,
  scalers, or label encoders are fit EXCLUSIVELY on the training set, and then 
  used to transform the test set. This is a vital industry practice.
- Label encoding: Categorical columns are converted into integer codes.
- StandardScaler: Fits to the numerical distribution and normalizes variables to 
  zero mean and unit variance, enabling distance-based models (like SVR/KNN) to perform well.
- Pure Python implementation: Completely isolated from the UI layer.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def clean_and_scale_data(df: pd.DataFrame, target_col: str, train_size: float = 0.8, random_state: int = 42):
    """
    Cleans, imputes, encodes, and scales features of a DataFrame for machine learning.
    
    Args:
        df (pd.DataFrame): Raw input dataset.
        target_col (str): Name of the target variable column.
        train_size (float): Proportion of the dataset to include in the train split.
        random_state (int): Controls shuffling applied to the split.
        
    Returns:
        tuple: (
            X_train_scaled,  # PD.DataFrame of training features (scaled)
            X_test_scaled,   # PD.DataFrame of testing features (scaled)
            y_train,         # Series of training target labels
            y_test,          # Series of testing target labels
            scaler,          # Fitted StandardScaler object (for playground predictions)
            feature_encoders,# Dict of fitted LabelEncoder objects for features
            target_encoder,  # Fitted LabelEncoder for target variable (or None)
            X_original,      # DataFrame of raw X features before preprocessing (for UI boundaries)
            num_cols,        # List of numerical feature column names
            cat_cols         # List of categorical feature column names
        )
    """
    df_clean = df.copy()
    
    # 1. Drop columns that are entirely null (these carry no information)
    df_clean = df_clean.dropna(how='all', axis=1)
    
    # 2. Extract features (X) and target (y)
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    # 3. Impute missing values in Target (y)
    if y.isnull().any():
        if not np.issubdtype(y.dtype, np.number) or y.dtype == 'object' or y.dtype.name == 'category':
            y_imputer = SimpleImputer(strategy='most_frequent')
            y_imputed = y_imputer.fit_transform(y.values.reshape(-1, 1)).ravel()
            y = pd.Series(y_imputed, name=target_col, index=df_clean.index)
        else:
            y_imputer = SimpleImputer(strategy='median')
            y_imputed = y_imputer.fit_transform(y.values.reshape(-1, 1)).ravel()
            y = pd.Series(y_imputed, name=target_col, index=df_clean.index)
            
    # 4. Encode target if it is categorical (string, object, category)
    target_encoder = None
    y_encoded = y.copy()
    if not np.issubdtype(y.dtype, np.number) or y.dtype == 'object' or y.dtype.name == 'category':
        target_encoder = LabelEncoder()
        y_encoded = pd.Series(target_encoder.fit_transform(y.astype(str)), name=target_col, index=df_clean.index)
        
    # 5. Split raw X and encoded y first to prevent data leakage during scaling/imputation!
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y_encoded, train_size=train_size, random_state=random_state
    )
    
    # 6. Identify column types
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Work on copies
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    
    # 7. Impute missing feature values
    # Numerical: Impute with training column mean
    if num_cols:
        num_imputer = SimpleImputer(strategy='mean')
        X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
        X_test[num_cols] = num_imputer.transform(X_test[num_cols])
        
    # Categorical: Impute with training column mode (most frequent)
    feature_encoders = {}
    if cat_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])
        
        # Label Encode categorical columns
        for col in cat_cols:
            le = LabelEncoder()
            # Fit and transform on training data
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            
            # Safely transform test data (handling unseen categories)
            test_vals = X_test[col].astype(str).tolist()
            train_classes = set(le.classes_)
            
            # Map unseen categories in the test set to the first class in the fitted LabelEncoder
            safe_test_vals = [val if val in train_classes else le.classes_[0] for val in test_vals]
            X_test[col] = le.transform(safe_test_vals)
            
            feature_encoders[col] = le
            
    # 8. Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reconstruct clean DataFrames with matching indexes and column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    return (
        X_train_scaled_df, 
        X_test_scaled_df, 
        y_train, 
        y_test, 
        scaler, 
        feature_encoders, 
        target_encoder, 
        X,  # Raw features before split and scale (useful for calculating min/max slider ranges in UI)
        num_cols, 
        cat_cols
    )
