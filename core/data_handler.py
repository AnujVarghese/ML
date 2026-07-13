"""
Data Handler Utility Engine (Data Layer)
----------------------------------------
This module is responsible for dataset ingest. It handles parsing user-uploaded 
files (CSV or Excel) and retrieving fallback built-in datasets from scikit-learn.

Design Rationale:
- Pure Python implementation: No Streamlit imports or GUI bindings, ensuring this 
  code can be reused in non-web scripts, notebooks, or automated pipelines.
- LRU Cache: Uses Python's standard `functools.lru_cache` to keep loaded datasets 
  in memory. Since fallback datasets don't change, caching avoids redundant disc/net I/O.
"""

import pandas as pd
from functools import lru_cache
from sklearn.datasets import fetch_california_housing, load_iris, load_wine

def load_custom_data(file_obj, file_name: str) -> pd.DataFrame:
    """
    Safely reads a user-uploaded CSV or Excel file into a Pandas DataFrame.
    
    Args:
        file_obj (file-like): The buffer containing raw uploaded data.
        file_name (str): The file name, used to extract the file extension.
        
    Returns:
        pd.DataFrame: Parsed tabular dataset.
    """
    # Check the extension to determine parsing engine
    if file_name.endswith('.csv'):
        return pd.read_csv(file_obj)
    elif file_name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_obj)
    else:
        raise ValueError("Invalid file format. Please upload a CSV or Excel file.")

@lru_cache(maxsize=8)
def load_fallback_data(dataset_name: str):
    """
    Retrieves a built-in scikit-learn dataset, returning it as a native 
    Pandas DataFrame along with its default target column name.
    
    Uses lru_cache to ensure that repeated calls are instantaneous.
    
    Args:
        dataset_name (str): The dataset identifier ('california_housing', 'iris', 'wine').
        
    Returns:
        tuple: (pd.DataFrame, str) representing the dataset and the target column name.
    """
    if dataset_name == 'california_housing':
        try:
            # fetch_california_housing fetches housing prices in California districts
            data = fetch_california_housing(as_frame=True)
            df = data.frame
            target = "MedHouseVal"
            return df, target
        except Exception:
            # Offline fallback: load diabetes dataset if no internet connection is available
            from sklearn.datasets import load_diabetes
            data = load_diabetes(as_frame=True)
            df = data.frame
            target = "target"
            return df, target
            
    elif dataset_name == 'iris':
        # load_iris fetches flower classification attributes
        data = load_iris(as_frame=True)
        df = data.frame
        # Map target numbers (0, 1, 2) to class labels for realistic preprocessing checks
        df['target'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        return df, "target"
        
    elif dataset_name == 'wine':
        # load_wine fetches wine chemical profiles
        data = load_wine(as_frame=True)
        df = data.frame
        # Map target classes to category labels
        df['target'] = df['target'].map({0: 'class_0', 1: 'class_1', 2: 'class_2'})
        return df, "target"
        
    else:
        raise ValueError(f"Unknown fallback dataset: {dataset_name}")
