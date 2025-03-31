#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Main pipeline for sleep data analysis and mood prediction.
# Handles data processing, feature extraction, model training, and evaluation in a single workflow.

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Define project directories
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# Create directories if they don't exist
for directory in [PROCESSED_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def load_data():
    # Loads raw sleep and mood data for processing
    logger.info("Loading sleep data")
    
    # Check for raw data file
    sleep_data_path = DATA_DIR / "raw" / "sleep_health_dataset.csv"
    if not sleep_data_path.exists():
        # Check for processed data as fallback
        processed_data_path = PROCESSED_DIR / "sleep_efficiency_processed.csv"
        if processed_data_path.exists():
            logger.info(f"Raw data not found, using processed data from {processed_data_path}")
            return pd.read_csv(processed_data_path)
        else:
            logger.error("No sleep data found. Please add data to the data directory.")
            return None
    
    # Load raw data
    try:
        df = pd.read_csv(sleep_data_path)
        logger.info(f"Loaded {len(df)} records from {sleep_data_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    # Preprocesses raw sleep data to handle missing values, outliers, and format conversion
    if df is None:
        return None
    
    logger.info("Preprocessing sleep data")
    
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Handle missing values
    if df_processed.isnull().sum().sum() > 0:
        logger.info(f"Found {df_processed.isnull().sum().sum()} missing values")
        
        # Fill numerical columns with mean
        numerical_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col].fillna(df_processed[col].mean(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    
    # Convert categorical variables to numeric if needed
    if 'Sleep Disorder' in df_processed.columns:
        # Create indicator variables for each disorder
        if df_processed['Sleep Disorder'].dtype == 'object':
            # Map sleep disorders to numeric values
            disorder_map = {
                'None': 0,
                'Sleep Apnea': 1,
                'Insomnia': 2
            }
            df_processed['Sleep Disorder'] = df_processed['Sleep Disorder'].map(disorder_map)
    
    # Convert duration to minutes if in hours
    if 'Sleep Duration' in df_processed.columns:
        if df_processed['Sleep Duration'].max() < 24:
            logger.info("Converting sleep duration from hours to minutes")
            df_processed['Sleep Duration'] = df_processed['Sleep Duration'] * 60
    
    # Create mood label if not present (for synthetic/demo purposes)
    if 'Mood' not in df_processed.columns and 'Sleep Quality' in df_processed.columns:
        logger.info("Creating synthetic mood label based on sleep quality")
        # Assume sleep quality ≥ 7 correlates with good mood
        df_processed['Mood'] = (df_processed['Sleep Quality'] >= 7).astype(int)
        df_processed['Mood'] = df_processed['Mood'].replace({1: 'Good', 0: 'Bad'})
    
    # Save processed data
    processed_file = PROCESSED_DIR / "sleep_efficiency_processed.csv"
    df_processed.to_csv(processed_file, index=False)
    logger.info(f"Saved processed data to {processed_file}")
    
    return df_processed

def extract_features(df_processed):
    # Extracts relevant features for sleep efficiency and mood prediction
    if df_processed is None:
        return None
    
    logger.info("Extracting features for mood prediction")
    
    # Define standard sleep metrics based on typical field names in sleep datasets
    sleep_metrics = [
        'Sleep Duration', 'Sleep Efficiency', 'REM Sleep Percentage',
        'Deep Sleep Percentage', 'Light Sleep Percentage', 'Awakenings',
        'Sleep Quality'
    ]
    
    # Filter to metrics that exist in this dataset
    available_metrics = [metric for metric in sleep_metrics if metric in df_processed.columns]
    
    if len(available_metrics) < 3:
        logger.warning("Too few sleep metrics available for feature extraction")
        # Try alternative field names
        common_alternatives = {
            'Sleep Duration': ['Total Sleep Time', 'TST', 'Sleep Time'],
            'Sleep Efficiency': ['Efficiency', 'Sleep Efficiency %'],
            'REM Sleep Percentage': ['REM %', 'REM Sleep %', 'REM'],
            'Deep Sleep Percentage': ['SWS %', 'Deep Sleep %', 'Deep'],
            'Light Sleep Percentage': ['Light %', 'Light Sleep %', 'Light'],
            'Awakenings': ['Wake Count', 'Number of Awakenings', 'Wakes'],
            'Sleep Quality': ['Quality', 'Sleep Score', 'Sleep Rating']
        }
        
        # Map alternative names to standard names
        for standard, alternatives in common_alternatives.items():
            if standard not in df_processed.columns:
                for alt in alternatives:
                    if alt in df_processed.columns:
                        df_processed[standard] = df_processed[alt]
                        logger.info(f"Mapped '{alt}' to '{standard}'")
                        if standard not in available_metrics:
                            available_metrics.append(standard)
                        break
    
    # Add lifestyle factors if available
    lifestyle_factors = [
        'Stress Level', 'Physical Activity Level', 'Exercise', 
        'Heart Rate', 'Caffeine Consumption', 'Alcohol Consumption'
    ]
    
    for factor in lifestyle_factors:
        if factor in df_processed.columns:
            if factor not in available_metrics:
                available_metrics.append(factor)
    
    # Check if we have a mood label
    if 'Mood' not in df_processed.columns:
        logger.error("No mood label found in the dataset")
        return None
    
    # Convert Mood to binary if it's categorical
    if df_processed['Mood'].dtype == 'object':
        mood_map = {'Good': 1, 'Bad': 0}
        df_processed['Mood'] = df_processed['Mood'].map(mood_map)
    
    # Create feature dataset
    X = df_processed[available_metrics]
    y = df_processed['Mood']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create DataFrame with scaled values
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Save features
    features = {
        'X_train': X_train_scaled_df,
        'y_train': y_train,
        'X_test': X_test_scaled_df,
        'y_test': y_test,
        'feature_names': list(X_train.columns),
        'scaler': scaler
    }
    
    features_file = FEATURES_DIR / "mood_prediction_features.joblib"
    joblib.dump(features, features_file)
    logger.info(f"Saved features to {features_file}")
    
    return features

def train_models(features):
    # Trains multiple models for mood prediction using extracted features
    if features is None:
        return None
    
    logger.info("Training mood prediction models")
    
    # Extract training data
    X_train = features['X_train']
    y_train = features['y_train']
    X_test = features['X_test']
    y_test = features['y_test']
    
    # Create model directory
    model_dir = MODELS_DIR / "mood_prediction"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Train Random Forest
    logger.info("Training Random Forest")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    # Evaluate Random Forest
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    
    metrics_rf = {
        'accuracy': float(accuracy_score(y_test, y_pred_rf)),
        'precision': float(precision_score(y_test, y_pred_rf)),
        'recall': float(recall_score(y_test, y_pred_rf)),
        'f1': float(f1_score(y_test, y_pred_rf)),
        'roc_auc': float(roc_auc_score(y_test, y_prob_rf))
    }
    
    logger.info(f"Random Forest metrics: {metrics_rf}")
    
    # Save Random Forest model
    rf_file = model_dir / "random_forest.joblib"
    rf_metadata_file = model_dir / "random_forest_metadata.joblib"
    
    rf_metadata = {
        'metrics': metrics_rf,
        'feature_names': features['feature_names'],
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 10
        }
    }
    
    joblib.dump(rf, rf_file)
    joblib.dump(rf_metadata, rf_metadata_file)
    logger.info(f"Saved Random Forest model to {rf_file}")
    
    # Train XGBoost
    logger.info("Training XGBoost")
    xgb_model = xgb.XGBClassifier(
        learning_rate=0.1,
        max_depth=4,
        n_estimators=100,
        subsample=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    
    # Evaluate XGBoost
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    metrics_xgb = {
        'accuracy': float(accuracy_score(y_test, y_pred_xgb)),
        'precision': float(precision_score(y_test, y_pred_xgb)),
        'recall': float(recall_score(y_test, y_pred_xgb)),
        'f1': float(f1_score(y_test, y_pred_xgb)),
        'roc_auc': float(roc_auc_score(y_test, y_prob_xgb))
    }
    
    logger.info(f"XGBoost metrics: {metrics_xgb}")
    
    # Save XGBoost model
    xgb_file = model_dir / "xgboost.joblib"
    xgb_metadata_file = model_dir / "xgboost_metadata.joblib"
    
    xgb_metadata = {
        'metrics': metrics_xgb,
        'feature_names': features['feature_names'],
        'hyperparameters': {
            'learning_rate': 0.1,
            'max_depth': 4,
            'n_estimators': 100,
            'subsample': 0.8
        }
    }
    
    joblib.dump(xgb_model, xgb_file)
    joblib.dump(xgb_metadata, xgb_metadata_file)
    logger.info(f"Saved XGBoost model to {xgb_file}")
    
    # Determine best model
    if metrics_rf['f1'] > metrics_xgb['f1']:
        best_model = "random_forest"
        best_metrics = metrics_rf
    else:
        best_model = "xgboost"
        best_metrics = metrics_xgb
    
    logger.info(f"Best model: {best_model} (F1: {best_metrics['f1']:.4f})")
    
    # Create symlink to best model
    best_model_file = model_dir / f"{best_model}.joblib"
    best_model_link = MODELS_DIR / "best_mood_model.joblib"
    
    try:
        if best_model_link.exists():
            best_model_link.unlink()
        
        best_model_link.symlink_to(best_model_file.relative_to(best_model_link.parent))
        logger.info(f"Created symlink to best model at {best_model_link}")
    except Exception as e:
        logger.error(f"Error creating symlink to best model: {e}")
    
    return {
        'random_forest': {
            'model': rf,
            'metrics': metrics_rf
        },
        'xgboost': {
            'model': xgb_model,
            'metrics': metrics_xgb
        },
        'best_model': best_model
    }

def run_pipeline():
    # Runs the complete pipeline from data loading to model training
    logger.info("Starting sleep analysis pipeline")
    
    # Load data
    data = load_data()
    if data is None:
        return False
    
    # Preprocess data
    processed_data = preprocess_data(data)
    if processed_data is None:
        return False
    
    # Extract features
    features = extract_features(processed_data)
    if features is None:
        return False
    
    # Train models
    models = train_models(features)
    if models is None:
        return False
    
    logger.info("Pipeline completed successfully")
    return True

def main():
    # Main function to execute the pipeline
    success = run_pipeline()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 