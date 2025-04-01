#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for running the baseline models with reduced parameter grids for faster testing.
This is useful for quickly testing if the models can be trained successfully.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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

# Define directories
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = ROOT_DIR / "models" / "baseline_comparison"
RESULTS_DIR = ROOT_DIR / "results" / "baseline_comparison"

def load_features():
    """
    Load features for mood prediction.
    
    Returns:
        dict: Dictionary containing features
    """
    features_file = FEATURES_DIR / "mood_prediction_features.joblib"
    
    if not features_file.exists():
        logger.error(f"Features file not found: {features_file}")
        return None
    
    try:
        features = joblib.load(features_file)
        logger.info(f"Loaded features from {features_file}")
        logger.info(f"Training set: {features['X_train'].shape}, Test set: {features['X_test'].shape}")
        return features
    
    except Exception as e:
        logger.error(f"Error loading features: {e}")
        return None

def train_and_evaluate_model(model_type, X_train, y_train, X_test, y_test):
    """
    Train and evaluate a single model.
    
    Args:
        model_type (str): Type of model to train ('logistic_regression', 'svm', 'random_forest', 'xgboost')
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
    
    Returns:
        tuple: (model, metrics, best_params)
    """
    logger.info(f"Training {model_type} model")
    
    # Define simplified parameter grids for faster testing
    if model_type == "logistic_regression":
        model = LogisticRegression(
            random_state=42,
            max_iter=2000
        )
        param_grid = {
            "C": [0.1, 1.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
            "class_weight": [None, "balanced"]
        }
        
    elif model_type == "svm":
        model = SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        )
        param_grid = {
            "C": [0.1, 10.0],
            "gamma": ["scale", 0.1],
            "class_weight": [None, "balanced"]
        }
        
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, None],
            "min_samples_split": [2, 5],
            "class_weight": [None, "balanced"]
        }
        
    elif model_type == "xgboost":
        model = xgb.XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0]
        }
        
    else:
        logger.error(f"Unknown model type: {model_type}")
        return None, None, None
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Perform grid search with reduced verbosity
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )
    
    # Train the model
    grid_search.fit(X_train, y_train)
    
    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    logger.info(f"Best parameters for {model_type}: {best_params}")
    logger.info(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
    
    # Make predictions on test set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }
    
    # Log results
    logger.info(f"Model evaluation metrics:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    
    return best_model, metrics, best_params

def main():
    """
    Main function to test a specific model.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test baseline model training.")
    parser.add_argument(
        "--model", 
        choices=["logistic_regression", "svm", "random_forest", "xgboost", "all"],
        default="all",
        help="Model to train and evaluate"
    )
    args = parser.parse_args()
    
    logger.info(f"Testing baseline model: {args.model}")
    
    # Load features
    features = load_features()
    if features is None:
        logger.error("Failed to load features. Exiting.")
        return 1
    
    # Extract data
    X_train = features["X_train"]
    y_train = features["y_train"]
    X_test = features["X_test"]
    y_test = features["y_test"]
    
    # Determine which models to train
    if args.model == "all":
        models_to_train = ["logistic_regression", "svm", "random_forest", "xgboost"]
    else:
        models_to_train = [args.model]
    
    # Train each model
    results = {}
    for model_type in models_to_train:
        logger.info(f"\n{'='*40}\nTraining {model_type}\n{'='*40}")
        model, metrics, params = train_and_evaluate_model(model_type, X_train, y_train, X_test, y_test)
        
        if model is not None:
            results[model_type] = {
                "metrics": metrics,
                "best_params": params
            }
    
    # Print summary
    logger.info("\n\n" + "="*50)
    logger.info("Model Performance Summary:")
    logger.info("="*50)
    
    for model_name, result in results.items():
        metrics = result["metrics"]
        logger.info(f"\n{model_name}:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    
    # Identify best model by F1 score
    if results:
        best_model = max(results.keys(), key=lambda k: results[k]["metrics"]["f1"])
        best_f1 = results[best_model]["metrics"]["f1"]
        logger.info("\n" + "="*50)
        logger.info(f"Best model: {best_model} with F1 score: {best_f1:.4f}")
        logger.info("="*50)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 