#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script creates mock data for testing the hyperparameter tuning process.
"""

import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.datasets import make_classification

# Define the root directory
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
FEATURES_DIR = DATA_DIR / "features"

# Ensure directories exist
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

def create_mock_data(task, n_samples=1000, n_features=20, random_state=42):
    """
    Create mock data for a specific task.
    
    Args:
        task (str): Task name ('rem_detection' or 'mood_prediction')
        n_samples (int): Number of samples
        n_features (int): Number of features
        random_state (int): Random state for reproducibility
    
    Returns:
        dict: Dictionary containing features and labels
    """
    print(f"Creating mock data for {task} with {n_samples} samples and {n_features} features...")
    
    # Create mock classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.8),
        n_redundant=int(n_features * 0.1),
        n_classes=2,
        random_state=random_state
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Convert to DataFrames
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    # Split into train and test sets (80/20)
    n_train = int(n_samples * 0.8)
    
    X_train = X_df.iloc[:n_train]
    y_train = y_series.iloc[:n_train]
    X_test = X_df.iloc[n_train:]
    y_test = y_series.iloc[n_train:]
    
    # Create a dictionary containing the data
    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
        "task_name": task,
        "n_samples": n_samples,
        "n_features": n_features
    }
    
    return data

def save_mock_data(task, data):
    """
    Save mock data to disk.
    
    Args:
        task (str): Task name
        data (dict): Dictionary containing the data
    """
    # Create file path
    file_path = FEATURES_DIR / f"{task}_features.joblib"
    
    # Save data
    joblib.dump(data, file_path)
    print(f"Saved mock data for {task} to {file_path}")

def main():
    """Main function to create mock data."""
    # Create mock data for REM detection
    rem_data = create_mock_data("rem_detection", n_samples=2000, n_features=30)
    save_mock_data("rem_detection", rem_data)
    
    # Create mock data for mood prediction
    mood_data = create_mock_data("mood_prediction", n_samples=1500, n_features=25)
    save_mock_data("mood_prediction", mood_data)
    
    print("Mock data creation complete.")

if __name__ == "__main__":
    main() 