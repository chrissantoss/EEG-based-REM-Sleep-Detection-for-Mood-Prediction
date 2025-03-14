#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script trains machine learning models for REM sleep detection and mood prediction.
It supports multiple model types and hyperparameter tuning.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

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
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = ROOT_DIR / "models"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Define model configurations
MODEL_CONFIGS = {
    "logistic_regression": {
        "model": LogisticRegression(random_state=42, max_iter=1000),
        "params": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
            "class_weight": [None, "balanced"]
        },
        "description": "Logistic Regression - A linear model for binary classification"
    },
    "svm": {
        "model": SVC(random_state=42, probability=True),
        "params": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto", 0.1, 0.01],
            "class_weight": [None, "balanced"]
        },
        "description": "Support Vector Machine - A powerful classifier that works well with non-linear data"
    },
    "random_forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": [None, "balanced"]
        },
        "description": "Random Forest - An ensemble of decision trees that works well with high-dimensional data"
    },
    "xgboost": {
        "model": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "scale_pos_weight": [1, 3, 5]
        },
        "description": "XGBoost - A gradient boosting algorithm that often achieves state-of-the-art results"
    }
}

def load_features(task):
    """
    Load extracted features for a specific task.
    
    Args:
        task (str): Task name ('rem_detection' or 'mood_prediction')
    
    Returns:
        dict: Dictionary containing features
    """
    # Determine the features file
    if task == "rem_detection":
        features_file = FEATURES_DIR / "rem_detection_features.joblib"
    elif task == "mood_prediction":
        features_file = FEATURES_DIR / "mood_prediction_features.joblib"
    else:
        logger.error(f"Unknown task: {task}")
        return None
    
    # Check if the file exists
    if not features_file.exists():
        logger.error(f"Features file not found: {features_file}")
        return None
    
    try:
        # Load the features
        features = joblib.load(features_file)
        logger.info(f"Loaded features for {task} from {features_file}")
        return features
    
    except Exception as e:
        logger.error(f"Error loading features from {features_file}: {e}")
        return None

def train_model(model_name, X_train, y_train, params=None, cv=5):
    """
    Train a machine learning model with hyperparameter tuning.
    
    Args:
        model_name (str): Name of the model to train
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        params (dict, optional): Hyperparameters to tune
        cv (int): Number of cross-validation folds
    
    Returns:
        object: Trained model
        dict: Best hyperparameters
    """
    # Check if the model is supported
    if model_name not in MODEL_CONFIGS:
        logger.error(f"Unsupported model: {model_name}")
        return None, None
    
    # Get model configuration
    model_config = MODEL_CONFIGS[model_name]
    model = model_config["model"]
    
    # Use default parameters if none provided
    if params is None:
        params = model_config["params"]
    
    logger.info(f"Training {model_name}: {model_config['description']}")
    
    # Set up cross-validation
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=cv_strategy,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )
    
    # Train the model
    grid_search.fit(X_train, y_train)
    
    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    logger.info(f"Best parameters for {model_name}: {best_params}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return best_model, best_params

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    
    Args:
        model (object): Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    
    # Log results
    logger.info(f"Evaluation metrics:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    
    # Print classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    return metrics

def save_model(model, model_name, task, metrics, params):
    """
    Save a trained model and its metadata.
    
    Args:
        model (object): Trained model
        model_name (str): Name of the model
        task (str): Task name
        metrics (dict): Evaluation metrics
        params (dict): Model hyperparameters
    
    Returns:
        Path: Path to the saved model
    """
    # Create task-specific directory
    task_dir = MODELS_DIR / task
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model file name
    model_file = task_dir / f"{model_name}.joblib"
    
    # Create metadata
    metadata = {
        "model_name": model_name,
        "task": task,
        "metrics": metrics,
        "params": params,
        "description": MODEL_CONFIGS[model_name]["description"]
    }
    
    # Save model and metadata
    try:
        joblib.dump(model, model_file)
        
        # Save metadata
        metadata_file = task_dir / f"{model_name}_metadata.joblib"
        joblib.dump(metadata, metadata_file)
        
        logger.info(f"Saved model to {model_file}")
        logger.info(f"Saved metadata to {metadata_file}")
        
        return model_file
    
    except Exception as e:
        logger.error(f"Error saving model to {model_file}: {e}")
        return None

def train_rem_detection_model(model_name):
    """
    Train a model for REM sleep detection.
    
    Args:
        model_name (str): Name of the model to train
    
    Returns:
        Path: Path to the saved model
    """
    # Load features
    features = load_features("rem_detection")
    
    if features is None:
        return None
    
    # Extract training and test data
    X_train = features["X_train"]
    y_train = features["y_train"]
    X_test = features["X_test"]
    y_test = features["y_test"]
    
    # Train the model
    model, params = train_model(model_name, X_train, y_train)
    
    if model is None:
        return None
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save the model
    model_file = save_model(model, model_name, "rem_detection", metrics, params)
    
    return model_file

def train_mood_prediction_model(model_name):
    """
    Train a model for mood prediction.
    
    Args:
        model_name (str): Name of the model to train
    
    Returns:
        Path: Path to the saved model
    """
    # Load features
    features = load_features("mood_prediction")
    
    if features is None:
        return None
    
    # Extract training and test data
    X_train = features["X_train"]
    y_train = features["y_train"]
    X_test = features["X_test"]
    y_test = features["y_test"]
    
    # Train the model
    model, params = train_model(model_name, X_train, y_train)
    
    if model is None:
        return None
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save the model
    model_file = save_model(model, model_name, "mood_prediction", metrics, params)
    
    return model_file

def main():
    """Main function to train models."""
    parser = argparse.ArgumentParser(description="Train machine learning models")
    parser.add_argument(
        "--task", 
        choices=["rem_detection", "mood_prediction", "all"],
        default="all",
        help="Training task (default: all)"
    )
    parser.add_argument(
        "--model", 
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        default="all",
        help="Model to train (default: all)"
    )
    args = parser.parse_args()
    
    # Determine which models to train
    if args.model == "all":
        models = list(MODEL_CONFIGS.keys())
    else:
        models = [args.model]
    
    logger.info(f"Starting training for task: {args.task}, models: {models}")
    
    # Train models for each task
    for model_name in models:
        if args.task in ["rem_detection", "all"]:
            logger.info(f"Training {model_name} for REM detection")
            train_rem_detection_model(model_name)
        
        if args.task in ["mood_prediction", "all"]:
            logger.info(f"Training {model_name} for mood prediction")
            train_mood_prediction_model(model_name)
    
    logger.info("Training completed")

if __name__ == "__main__":
    main() 