#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script extracts features from processed EEG data for model training.
It combines features from multiple datasets and prepares them for machine learning.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Define the data directory
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

# Ensure directories exist
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

def load_processed_data(dataset_name):
    """
    Load processed data for a specific dataset.
    
    Args:
        dataset_name (str): Name of the dataset
    
    Returns:
        pd.DataFrame: DataFrame containing processed data
    """
    dataset_dir = PROCESSED_DIR / dataset_name
    
    if not dataset_dir.exists():
        logger.error(f"Processed data directory not found: {dataset_dir}")
        return None
    
    # Find all CSV files in the directory
    csv_files = list(dataset_dir.glob("*.csv"))
    
    if not csv_files:
        logger.error(f"No processed data files found in {dataset_dir}")
        return None
    
    # Load and concatenate all files
    dfs = []
    
    for file in tqdm(csv_files, desc=f"Loading {dataset_name} files"):
        try:
            df = pd.read_csv(file)
            
            # Add dataset source column
            df['dataset'] = dataset_name
            df['file'] = file.name
            
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    
    if not dfs:
        logger.error(f"Failed to load any data from {dataset_name}")
        return None
    
    # Concatenate all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined_df)} samples from {dataset_name}")
    
    return combined_df

def select_features(X, y, method='anova', k=20):
    """
    Select the most informative features.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        method (str): Feature selection method ('anova' or 'mutual_info')
        k (int): Number of features to select
    
    Returns:
        pd.DataFrame: Selected features
        list: Names of selected features
    """
    # Choose the scoring function
    if method == 'anova':
        score_func = f_classif
    elif method == 'mutual_info':
        score_func = mutual_info_classif
    else:
        logger.error(f"Unknown feature selection method: {method}")
        return X, list(X.columns)
    
    # Apply feature selection
    selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = [X.columns[i] for i in selected_indices]
    
    logger.info(f"Selected {len(selected_features)} features using {method}")
    
    # Convert back to DataFrame
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features)
    
    return X_selected_df, selected_features

def prepare_rem_detection_features(data):
    """
    Prepare features for REM sleep detection.
    
    Args:
        data (pd.DataFrame): Combined processed data
    
    Returns:
        dict: Dictionary containing train/test splits for REM detection
    """
    # Check if sleep stage column exists
    if 'sleep_stage' not in data.columns:
        logger.error("Sleep stage column not found in data")
        return None
    
    # Create binary target: REM vs. Non-REM
    data['is_rem'] = (data['sleep_stage'] == 4).astype(int)
    
    # Select features (exclude non-feature columns)
    non_feature_cols = ['segment_idx', 'sleep_stage', 'sleep_stage_name', 
                        'dataset', 'file', 'is_rem']
    feature_cols = [col for col in data.columns if col not in non_feature_cols]
    
    # Handle missing values
    data[feature_cols] = data[feature_cols].fillna(0)
    
    # Split data
    X = data[feature_cols]
    y = data['is_rem']
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Select most informative features
    X_selected, selected_features = select_features(X_scaled, y, method='anova', k=30)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create a dictionary with the results
    result = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'selected_features': selected_features,
        'scaler': scaler,
        'feature_cols': feature_cols
    }
    
    logger.info(f"Prepared REM detection features: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    return result

def prepare_mood_prediction_features(data):
    """
    Prepare features for mood prediction.
    
    Args:
        data (pd.DataFrame): Sleep-mood dataset
    
    Returns:
        dict: Dictionary containing train/test splits for mood prediction
    """
    # Check if mood rating column exists
    if 'mood_rating' not in data.columns:
        logger.error("Mood rating column not found in data")
        return None
    
    # Create binary target: Good mood (>6) vs. Bad mood (<=6)
    data['good_mood'] = (data['mood_rating'] > 6).astype(int)
    
    # Select features (exclude non-feature columns)
    non_feature_cols = ['subject_id', 'night', 'date', 'dataset', 'file',
                        'mood_rating', 'anxiety_rating', 'good_mood']
    feature_cols = [col for col in data.columns if col not in non_feature_cols]
    
    # Handle missing values
    data[feature_cols] = data[feature_cols].fillna(0)
    
    # Split data
    X = data[feature_cols]
    y = data['good_mood']
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Select most informative features
    X_selected, selected_features = select_features(X_scaled, y, method='mutual_info', k=10)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create a dictionary with the results
    result = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'selected_features': selected_features,
        'scaler': scaler,
        'feature_cols': feature_cols
    }
    
    logger.info(f"Prepared mood prediction features: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    return result

def save_features(features_dict, output_file):
    """
    Save extracted features to a file.
    
    Args:
        features_dict (dict): Dictionary containing features
        output_file (Path): Path to save the features
    
    Returns:
        bool: True if successful, False otherwise
    """
    import joblib
    
    try:
        # Create directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the features dictionary
        joblib.dump(features_dict, output_file)
        
        logger.info(f"Saved features to {output_file}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving features to {output_file}: {e}")
        return False

def extract_rem_detection_features():
    """
    Extract features for REM sleep detection.
    
    Returns:
        Path: Path to the saved features file
    """
    # Load processed data from EEG datasets
    eeg_datasets = ["sleep-edf", "sleep-cassette"]
    dfs = []
    
    for dataset_name in eeg_datasets:
        df = load_processed_data(dataset_name)
        if df is not None:
            dfs.append(df)
    
    if not dfs:
        logger.error("No data loaded for REM detection")
        return None
    
    # Combine all datasets
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined {len(combined_df)} samples for REM detection")
    
    # Prepare features
    features_dict = prepare_rem_detection_features(combined_df)
    
    if features_dict is None:
        return None
    
    # Save features
    output_file = FEATURES_DIR / "rem_detection_features.joblib"
    save_features(features_dict, output_file)
    
    return output_file

def extract_mood_prediction_features():
    """
    Extract features for mood prediction.
    
    Returns:
        Path: Path to the saved features file
    """
    # Load processed data from sleep-mood dataset
    df = load_processed_data("sleep-mood")
    
    if df is None:
        logger.error("No data loaded for mood prediction")
        return None
    
    # Prepare features
    features_dict = prepare_mood_prediction_features(df)
    
    if features_dict is None:
        return None
    
    # Save features
    output_file = FEATURES_DIR / "mood_prediction_features.joblib"
    save_features(features_dict, output_file)
    
    return output_file

def main():
    """Main function to extract features."""
    parser = argparse.ArgumentParser(description="Extract features from processed EEG data")
    parser.add_argument(
        "--task", 
        choices=["rem_detection", "mood_prediction", "all"],
        default="all",
        help="Feature extraction task (default: all)"
    )
    args = parser.parse_args()
    
    logger.info(f"Starting feature extraction for task: {args.task}")
    
    if args.task in ["rem_detection", "all"]:
        extract_rem_detection_features()
    
    if args.task in ["mood_prediction", "all"]:
        extract_mood_prediction_features()
    
    logger.info("Feature extraction completed")

if __name__ == "__main__":
    main() 