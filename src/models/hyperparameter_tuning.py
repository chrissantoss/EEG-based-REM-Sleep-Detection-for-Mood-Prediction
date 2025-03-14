#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script provides advanced hyperparameter tuning capabilities for machine learning models.
It tracks performance before and after tuning and saves the results for comparison.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import joblib
import json
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from scipy.stats import uniform, randint, loguniform

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
TUNING_DIR = ROOT_DIR / "models" / "tuning_results"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TUNING_DIR.mkdir(parents=True, exist_ok=True)

# Define extended parameter distributions for randomized search
PARAM_DISTRIBUTIONS = {
    "logistic_regression": {
        "C": loguniform(1e-4, 1e2),
        "penalty": ["l1", "l2", "elasticnet", "none"],
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "class_weight": [None, "balanced"],
        "max_iter": randint(500, 2000),
        "tol": loguniform(1e-6, 1e-3)
    },
    "svm": {
        "C": loguniform(1e-3, 1e3),
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "gamma": loguniform(1e-5, 1e1),
        "degree": randint(2, 6),
        "class_weight": [None, "balanced"],
        "tol": loguniform(1e-6, 1e-3),
        "coef0": uniform(0, 1)
    },
    "random_forest": {
        "n_estimators": randint(50, 500),
        "max_depth": [None] + list(randint(5, 50).rvs(5)),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", None] + list(uniform(0.1, 0.9).rvs(3)),
        "bootstrap": [True, False],
        "class_weight": [None, "balanced", "balanced_subsample"],
        "criterion": ["gini", "entropy", "log_loss"]
    },
    "xgboost": {
        "n_estimators": randint(50, 500),
        "max_depth": randint(3, 15),
        "learning_rate": loguniform(1e-3, 0.5),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "colsample_bylevel": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 10),
        "gamma": uniform(0, 1),
        "alpha": loguniform(1e-3, 10),
        "lambda": loguniform(1e-3, 10),
        "scale_pos_weight": uniform(1, 10)
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

def load_model(model_name, task):
    """
    Load a trained model and its metadata.
    
    Args:
        model_name (str): Name of the model
        task (str): Task name
    
    Returns:
        tuple: (model, metadata) or (None, None) if loading fails
    """
    # Determine file paths
    task_dir = MODELS_DIR / task
    model_file = task_dir / f"{model_name}.joblib"
    metadata_file = task_dir / f"{model_name}_metadata.joblib"
    
    # Check if files exist
    if not model_file.exists() or not metadata_file.exists():
        logger.error(f"Model or metadata file not found: {model_file} / {metadata_file}")
        return None, None
    
    try:
        # Load model and metadata
        model = joblib.load(model_file)
        metadata = joblib.load(metadata_file)
        
        logger.info(f"Loaded model from {model_file}")
        logger.info(f"Loaded metadata from {metadata_file}")
        
        return model, metadata
    
    except Exception as e:
        logger.error(f"Error loading model or metadata: {e}")
        return None, None

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
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    
    # Log results
    logger.info(f"Evaluation metrics:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics

def tune_hyperparameters(model_name, X_train, y_train, X_test, y_test, n_iter=50, cv=5, scoring="f1"):
    """
    Tune hyperparameters for a model using RandomizedSearchCV.
    
    Args:
        model_name (str): Name of the model to tune
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        n_iter (int): Number of parameter settings sampled
        cv (int): Number of cross-validation folds
        scoring (str): Scoring metric for hyperparameter tuning
    
    Returns:
        tuple: (tuned_model, best_params, tuning_results)
    """
    # Get base model
    if model_name == "logistic_regression":
        base_model = LogisticRegression(random_state=42)
    elif model_name == "svm":
        base_model = SVC(random_state=42, probability=True)
    elif model_name == "random_forest":
        base_model = RandomForestClassifier(random_state=42)
    elif model_name == "xgboost":
        base_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
    else:
        logger.error(f"Unsupported model: {model_name}")
        return None, None, None
    
    # Get parameter distributions
    param_dist = PARAM_DISTRIBUTIONS[model_name]
    
    # Set up cross-validation
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Evaluate base model before tuning
    base_model.fit(X_train, y_train)
    before_metrics = evaluate_model(base_model, X_test, y_test)
    logger.info(f"Performance before tuning: F1 = {before_metrics['f1']:.4f}")
    
    # Start timing
    start_time = time.time()
    
    # Perform randomized search
    logger.info(f"Starting hyperparameter tuning for {model_name} with {n_iter} iterations...")
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=-1,
        verbose=2,
        random_state=42,
        return_train_score=True
    )
    
    # Train the model
    random_search.fit(X_train, y_train)
    
    # End timing
    tuning_time = time.time() - start_time
    
    # Get the best model and parameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    # Evaluate tuned model
    after_metrics = evaluate_model(best_model, X_test, y_test)
    logger.info(f"Performance after tuning: F1 = {after_metrics['f1']:.4f}")
    
    # Calculate improvement
    f1_improvement = after_metrics['f1'] - before_metrics['f1']
    improvement_percent = (f1_improvement / before_metrics['f1']) * 100 if before_metrics['f1'] > 0 else float('inf')
    
    logger.info(f"F1 score improvement: {f1_improvement:.4f} ({improvement_percent:.2f}%)")
    
    # Compile tuning results
    tuning_results = {
        "model_name": model_name,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "best_params": best_params,
        "cv_results": {
            "mean_test_score": float(random_search.cv_results_["mean_test_score"][random_search.best_index_]),
            "std_test_score": float(random_search.cv_results_["std_test_score"][random_search.best_index_]),
            "mean_train_score": float(random_search.cv_results_["mean_train_score"][random_search.best_index_]),
            "std_train_score": float(random_search.cv_results_["std_train_score"][random_search.best_index_])
        },
        "improvement": {
            "f1_absolute": float(f1_improvement),
            "f1_percent": float(improvement_percent)
        },
        "tuning_time": float(tuning_time),
        "n_iter": n_iter,
        "cv": cv,
        "scoring": scoring,
        "timestamp": datetime.now().isoformat()
    }
    
    return best_model, best_params, tuning_results

def save_tuned_model(model, model_name, task, tuning_results):
    """
    Save a tuned model and its tuning results.
    
    Args:
        model (object): Tuned model
        model_name (str): Name of the model
        task (str): Task name
        tuning_results (dict): Results from hyperparameter tuning
    
    Returns:
        tuple: (model_path, results_path)
    """
    # Create task-specific directories
    task_dir = MODELS_DIR / task
    task_dir.mkdir(parents=True, exist_ok=True)
    
    tuning_task_dir = TUNING_DIR / task
    tuning_task_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create file paths
    model_file = task_dir / f"{model_name}_tuned_{timestamp}.joblib"
    results_file = tuning_task_dir / f"{model_name}_tuning_results_{timestamp}.json"
    
    try:
        # Save model
        joblib.dump(model, model_file)
        logger.info(f"Saved tuned model to {model_file}")
        
        # Save tuning results
        with open(results_file, 'w') as f:
            json.dump(tuning_results, f, indent=2)
        logger.info(f"Saved tuning results to {results_file}")
        
        return model_file, results_file
    
    except Exception as e:
        logger.error(f"Error saving tuned model or results: {e}")
        return None, None

def tune_model_for_task(model_name, task, n_iter=50, cv=5, scoring="f1"):
    """
    Tune hyperparameters for a specific model and task.
    
    Args:
        model_name (str): Name of the model to tune
        task (str): Task name
        n_iter (int): Number of parameter settings sampled
        cv (int): Number of cross-validation folds
        scoring (str): Scoring metric for hyperparameter tuning
    
    Returns:
        tuple: (model_path, results_path)
    """
    # Load features
    features = load_features(task)
    
    if features is None:
        return None, None
    
    # Extract training and test data
    X_train = features["X_train"]
    y_train = features["y_train"]
    X_test = features["X_test"]
    y_test = features["y_test"]
    
    # Load existing model if available
    existing_model, existing_metadata = load_model(model_name, task)
    
    if existing_model is not None:
        logger.info(f"Found existing {model_name} model for {task}")
        logger.info(f"Current performance: F1 = {existing_metadata['metrics']['f1']:.4f}")
    else:
        logger.info(f"No existing {model_name} model found for {task}")
    
    # Tune hyperparameters
    tuned_model, best_params, tuning_results = tune_hyperparameters(
        model_name, X_train, y_train, X_test, y_test, n_iter, cv, scoring
    )
    
    if tuned_model is None:
        return None, None
    
    # Save tuned model and results
    model_path, results_path = save_tuned_model(tuned_model, model_name, task, tuning_results)
    
    return model_path, results_path

def main():
    """Main function to tune hyperparameters."""
    parser = argparse.ArgumentParser(description="Tune hyperparameters for machine learning models")
    parser.add_argument(
        "--task", 
        choices=["rem_detection", "mood_prediction", "all"],
        default="all",
        help="Task to tune models for (default: all)"
    )
    parser.add_argument(
        "--model", 
        choices=list(PARAM_DISTRIBUTIONS.keys()) + ["all"],
        default="all",
        help="Model to tune (default: all)"
    )
    parser.add_argument(
        "--n_iter", 
        type=int,
        default=50,
        help="Number of parameter settings to sample (default: 50)"
    )
    parser.add_argument(
        "--cv", 
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--scoring", 
        choices=["accuracy", "precision", "recall", "f1", "roc_auc"],
        default="f1",
        help="Scoring metric for hyperparameter tuning (default: f1)"
    )
    args = parser.parse_args()
    
    # Determine which models to tune
    if args.model == "all":
        models = list(PARAM_DISTRIBUTIONS.keys())
    else:
        models = [args.model]
    
    # Determine which tasks to tune for
    if args.task == "all":
        tasks = ["rem_detection", "mood_prediction"]
    else:
        tasks = [args.task]
    
    logger.info(f"Starting hyperparameter tuning for tasks: {tasks}, models: {models}")
    logger.info(f"Using {args.n_iter} iterations, {args.cv} CV folds, and {args.scoring} scoring")
    
    # Tune models for each task
    for task in tasks:
        for model_name in models:
            logger.info(f"Tuning {model_name} for {task}")
            model_path, results_path = tune_model_for_task(
                model_name, task, args.n_iter, args.cv, args.scoring
            )
            
            if model_path is not None:
                logger.info(f"Successfully tuned {model_name} for {task}")
                logger.info(f"Model saved to: {model_path}")
                logger.info(f"Results saved to: {results_path}")
            else:
                logger.error(f"Failed to tune {model_name} for {task}")
    
    logger.info("Hyperparameter tuning completed")

if __name__ == "__main__":
    main() 