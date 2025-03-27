#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script performs advanced hyperparameter tuning specifically for the mood prediction task.
It includes a more refined hyperparameter search space and additional evaluation metrics.
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
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
MODELS_DIR = ROOT_DIR / "models"
TUNING_DIR = ROOT_DIR / "models" / "tuning_results" / "mood_prediction"
TUNING_DIR.mkdir(parents=True, exist_ok=True)

# Define an expanded hyperparameter grid for XGBoost
XGBOOST_PARAM_GRID = {
    # Tree parameters
    "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],  # Expanded depth options
    "min_child_weight": [1, 2, 3, 5, 7, 10],  # More options for min child weight
    "gamma": [0, 0.1, 0.2, 0.3, 0.5, 1.0],    # Expanded gamma values for pruning
    
    # Boosting parameters
    "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1, 0.2],  # More learning rate options
    "n_estimators": [100, 200, 300, 400, 500, 700, 1000],  # More estimators options
    
    # Sampling parameters
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Expanded sampling rates
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Expanded column sampling
    "colsample_bylevel": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Column sampling by level
    
    # Regularization parameters
    "reg_alpha": [0, 0.001, 0.01, 0.1, 1.0, 10.0],  # Expanded L1 regularization
    "reg_lambda": [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0],  # Expanded L2 regularization
    
    # Class imbalance parameters
    "scale_pos_weight": [1, 2, 3, 5, 7, 10]  # Expanded values for class imbalance
}

def load_features(task="mood_prediction"):
    """
    Load extracted features for the mood prediction task.
    
    Returns:
        dict: Dictionary containing features
    """
    # Try to load real data first
    try:
        from test_all_models import load_real_mood_data
        features = load_real_mood_data()
        if features is not None:
            logger.info("Successfully loaded real mood data")
            return features
    except Exception as e:
        logger.warning(f"Could not load real mood data: {e}")
    
    # Fall back to features file
    features_file = FEATURES_DIR / f"{task}_features.joblib"
    if not features_file.exists():
        logger.error(f"Features file not found: {features_file}")
        return None
    
    try:
        features = joblib.load(features_file)
        logger.info(f"Loaded features from {features_file}")
        return features
    except Exception as e:
        logger.error(f"Error loading features: {e}")
        return None

def load_existing_model(model_name="xgboost"):
    """
    Load an existing trained model.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        tuple: (model, metadata) or (None, None) if loading fails
    """
    # Try loading the refined model first
    model_path = MODELS_DIR / "mood_prediction" / f"{model_name}_refined.joblib"
    metadata_path = MODELS_DIR / "mood_prediction" / f"{model_name}_refined_metadata.joblib"
    
    # If refined model doesn't exist, try the base model
    if not model_path.exists() or not metadata_path.exists():
        model_path = MODELS_DIR / "mood_prediction" / f"{model_name}.joblib"
        metadata_path = MODELS_DIR / "mood_prediction" / f"{model_name}_metadata.joblib"
        
        if not model_path.exists() or not metadata_path.exists():
            logger.error(f"Model files not found: {model_path}, {metadata_path}")
            return None, None
    
    try:
        model = joblib.load(model_path)
        metadata = joblib.load(metadata_path)
        logger.info(f"Loaded model from {model_path}")
        return model, metadata
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

def evaluate_model(model, X, y):
    """
    Evaluate a model's performance.
    
    Args:
        model: Trained model
        X: Features
        y: Target labels
        
    Returns:
        dict: Evaluation metrics
    """
    # Handle feature mismatch - check if model has feature_names attribute
    if hasattr(model, 'feature_names_') and model.feature_names_ is not None:
        # Get the model's expected features
        model_features = model.feature_names_
        
        # Check if there are missing features in X
        missing_features = [f for f in model_features if f not in X.columns]
        
        if missing_features:
            logger.warning(f"Missing features in test data: {missing_features}")
            # Add missing features with zeros
            for feature in missing_features:
                X[feature] = 0
            
        # Ensure columns are in the same order as the model expects
        X = X[model_features]
    
    # Predict
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist()
    }
    
    return metrics

def tune_xgboost_hyperparameters(X_train, y_train, X_test, y_test, n_iter=50, cv=5):
    """
    Tune XGBoost hyperparameters with RandomizedSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        n_iter: Number of parameter settings sampled
        cv: Number of cross-validation folds
        
    Returns:
        tuple: (best_model, best_params, evaluation_metrics)
    """
    logger.info("Starting hyperparameter tuning for XGBoost...")
    start_time = time.time()
    
    # Create a baseline model with default parameters
    logger.info("Training baseline model with default parameters...")
    base_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    base_model.fit(X_train, y_train)
    
    # Evaluate the baseline model
    logger.info("Evaluating baseline model...")
    base_metrics = evaluate_model(base_model, X_test, y_test)
    logger.info(f"Baseline model metrics:")
    logger.info(f"  F1 score: {base_metrics['f1']:.4f}")
    logger.info(f"  Accuracy: {base_metrics['accuracy']:.4f}")
    logger.info(f"  ROC AUC: {base_metrics['roc_auc']:.4f}")
    
    # Set up cross-validation with stratification
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Use randomized search with multiple scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    # Create base model for tuning
    model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss'
    )
    
    # Create randomized search
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=XGBOOST_PARAM_GRID,
        n_iter=n_iter,
        scoring='f1',
        refit=True,
        cv=cv_strategy,
        verbose=2,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    # Fit the model
    logger.info(f"Fitting model with {n_iter} parameter combinations...")
    random_search.fit(X_train, y_train)
    
    # Get the best model and parameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    logger.info(f"Best CV score: {best_score:.4f}")
    logger.info("Best parameters:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")
    
    # Evaluate tuned model
    logger.info("Evaluating tuned model on test set...")
    tuned_metrics = evaluate_model(best_model, X_test, y_test)
    
    # Calculate improvements
    improvements = {}
    for metric in base_metrics.keys():
        if metric != 'confusion_matrix':
            improvements[metric] = tuned_metrics[metric] - base_metrics[metric]
            improvements[f"{metric}_pct"] = (improvements[metric] / max(base_metrics[metric], 0.001)) * 100
    
    logger.info(f"Tuned model F1 score: {tuned_metrics['f1']:.4f} (change: {improvements['f1']:.4f}, {improvements['f1_pct']:.2f}%)")
    logger.info(f"Tuned model ROC AUC: {tuned_metrics['roc_auc']:.4f} (change: {improvements['roc_auc']:.4f}, {improvements['roc_auc_pct']:.2f}%)")
    
    # Save the results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"xgboost_tuned_{timestamp}"
    model_path = MODELS_DIR / "mood_prediction" / f"{model_name}.joblib"
    metadata_path = MODELS_DIR / "mood_prediction" / f"{model_name}_metadata.joblib"
    
    # Prepare metadata
    metadata = {
        'model_name': model_name,
        'task': 'mood_prediction',
        'tuning_timestamp': timestamp,
        'metrics': tuned_metrics,
        'improvements': improvements,
        'params': best_params,
        'cv_score': best_score,
        'description': 'XGBoost model with advanced hyperparameter tuning'
    }
    
    # Save tuning results
    tuning_results = {
        'model_name': 'xgboost',
        'task': 'mood_prediction',
        'timestamp': timestamp,
        'before_metrics': base_metrics,
        'after_metrics': tuned_metrics,
        'improvements': improvements,
        'best_params': best_params,
        'cv_score': best_score,
        'tuning_time': time.time() - start_time
    }
    
    tuning_file = TUNING_DIR / f"xgboost_tuning_results_{timestamp}.json"
    with open(tuning_file, 'w') as f:
        json.dump(tuning_results, f, indent=2)
    
    # Save model and metadata
    joblib.dump(best_model, model_path)
    joblib.dump(metadata, metadata_path)
    
    logger.info(f"Saved tuned model to {model_path}")
    logger.info(f"Saved tuning results to {tuning_file}")
    
    return best_model, best_params, tuned_metrics

def fine_tune_specific_params(X_train, y_train, X_test, y_test, base_params):
    """
    Fine-tune specific parameters with a more focused search.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        base_params: Base parameters to start with
        
    Returns:
        tuple: (best_model, best_params, evaluation_metrics)
    """
    logger.info("Fine-tuning specific parameters with narrow search...")
    
    # Start with the base parameters
    best_params = base_params.copy()
    
    # Create a more focused parameter grid for learning rate
    param_grid = {
        "learning_rate": [
            max(0.001, best_params["learning_rate"] - 0.03), 
            max(0.005, best_params["learning_rate"] - 0.02),
            max(0.01, best_params["learning_rate"] - 0.01),
            best_params["learning_rate"],
            min(0.5, best_params["learning_rate"] + 0.01),
            min(0.5, best_params["learning_rate"] + 0.02),
            min(0.5, best_params["learning_rate"] + 0.03)
        ]
    }
    
    # Fine-tune with a grid search
    model = xgb.XGBClassifier(
        **best_params,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Set up cross-validation
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Find the best learning rate
    best_score = -1
    best_lr = best_params["learning_rate"]
    
    for lr in param_grid["learning_rate"]:
        model.learning_rate = lr
        scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='f1')
        avg_score = scores.mean()
        
        logger.info(f"Learning rate: {lr}, CV F1: {avg_score:.4f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_lr = lr
    
    best_params["learning_rate"] = best_lr
    logger.info(f"Best fine-tuned learning rate: {best_lr}")
    
    # Train the final model with the best params
    best_model = xgb.XGBClassifier(
        **best_params,
        random_state=42,
        eval_metric='logloss'
    )
    
    best_model.fit(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(best_model, X_test, y_test)
    logger.info(f"Fine-tuned model F1 score: {metrics['f1']:.4f}")
    
    # Save the fine-tuned model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"xgboost_finetuned_{timestamp}"
    model_path = MODELS_DIR / "mood_prediction" / f"{model_name}.joblib"
    metadata_path = MODELS_DIR / "mood_prediction" / f"{model_name}_metadata.joblib"
    
    # Prepare metadata
    metadata = {
        'model_name': model_name,
        'task': 'mood_prediction',
        'tuning_timestamp': timestamp,
        'metrics': metrics,
        'params': best_params,
        'cv_score': best_score,
        'description': 'XGBoost model with fine-tuned hyperparameters'
    }
    
    # Save model and metadata
    joblib.dump(best_model, model_path)
    joblib.dump(metadata, metadata_path)
    
    logger.info(f"Saved fine-tuned model to {model_path}")
    
    return best_model, best_params, metrics

def main():
    """Main function to tune XGBoost hyperparameters for mood prediction."""
    logger.info("Starting advanced mood prediction hyperparameter tuning")
    
    # Load features
    features = load_features()
    
    if features is None:
        logger.error("Failed to load features")
        return 1
    
    # Extract training and test data
    X_train = features["X_train"]
    y_train = features["y_train"]
    X_test = features["X_test"]
    y_test = features["y_test"]
    
    logger.info(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    logger.info(f"Features: {', '.join(X_train.columns)}")
    
    # Tune XGBoost hyperparameters
    best_model, best_params, tuned_metrics = tune_xgboost_hyperparameters(
        X_train, y_train, X_test, y_test, n_iter=50, cv=5
    )
    
    # Fine-tune specific parameters
    fine_tuned_model, fine_tuned_params, fine_tuned_metrics = fine_tune_specific_params(
        X_train, y_train, X_test, y_test, best_params
    )
    
    # Print final results
    logger.info("\n=== Final Results ===")
    improvement_pct = ((fine_tuned_metrics['f1'] - tuned_metrics['f1']) / tuned_metrics['f1']) * 100
    logger.info(f"Tuned model F1 score: {tuned_metrics['f1']:.4f}")
    logger.info(f"Fine-tuned model F1 score: {fine_tuned_metrics['f1']:.4f} (improvement: {improvement_pct:.2f}%)")
    logger.info(f"Fine-tuned parameters: {fine_tuned_params}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 