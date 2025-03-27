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
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  # More focused range
        "penalty": ["l1", "l2", "elasticnet"],  # Fixed penalty options (removed "none")
        "solver": ["newton-cg", "lbfgs", "liblinear", "saga"],  # Focus on better solvers
        "class_weight": [None, "balanced"],
        "max_iter": [1000, 2000, 3000],  # Common iteration values
        "tol": [1e-6, 1e-5, 1e-4],  # More focused tolerance values
        "l1_ratio": [0.1, 0.5, 0.9]  # For elasticnet penalty
    },
    "svm": {
        "C": [0.1, 1.0, 10.0, 100.0],  # More focused range
        "kernel": ["linear", "rbf", "poly"],  # Most common kernels
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],  # Better gamma options
        "degree": [2, 3, 4],  # For polynomial kernel
        "class_weight": [None, "balanced"],
        "probability": [True],  # Always True to get probability estimates
        "tol": [1e-5, 1e-4, 1e-3],  # More focused tolerance values
        "coef0": [0.0, 0.5, 1.0]  # For polynomial kernel
    },
    "random_forest": {
        "n_estimators": randint(100, 500),  # More focused range
        "max_depth": [None, 10, 20, 30, 40, 50],  # More focused depth options
        "min_samples_split": [2, 5, 10],  # Common values
        "min_samples_leaf": [1, 2, 4],  # Common values
        "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],  # Better feature selection options
        "bootstrap": [True],  # Bootstrap is typically better
        "class_weight": [None, "balanced"],  # Class weight options
        "criterion": ["gini", "entropy"],  # Simpler options
        "min_impurity_decrease": [0.0, 0.01, 0.05],  # More focused values
        "oob_score": [True, False]  # Add out-of-bag scoring
    },
    "xgboost": {
        "n_estimators": [50, 100, 200, 300, 400],  # More focused range
        "max_depth": [3, 4, 5, 6, 7, 8, 10],  # More focused depth options
        "learning_rate": [0.01, 0.05, 0.1, 0.2],  # Common learning rates
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],  # Better sampling rates
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],  # Better column sampling
        "min_child_weight": [1, 3, 5, 7],  # More focused values
        "gamma": [0, 0.1, 0.2, 0.3, 0.5],  # More focused values
        "reg_alpha": [0, 0.1, 1.0, 10.0],  # Better L1 regularization
        "reg_lambda": [0.1, 1.0, 10.0, 100.0],  # Better L2 regularization
        "scale_pos_weight": [1, 3, 5]  # Simpler values for class imbalance
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

def get_param_combinations(model_name):
    """
    Get compatible parameter combinations for a model.
    
    Args:
        model_name (str): Name of the model
    
    Returns:
        list: List of compatible parameter dictionaries
    """
    if model_name == "logistic_regression":
        # Create compatible solver/penalty combinations
        combinations = []
        
        # For saga solver (supports all penalties)
        for penalty in ["l1", "l2", "elasticnet"]:
            for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
                for class_weight in [None, "balanced"]:
                    for max_iter in [1000, 2000, 3000]:
                        for tol in [1e-6, 1e-5, 1e-4]:
                            if penalty == "elasticnet":
                                for l1_ratio in [0.1, 0.5, 0.9]:
                                    combinations.append({
                                        "penalty": penalty,
                                        "solver": "saga",
                                        "C": C,
                                        "class_weight": class_weight,
                                        "max_iter": max_iter,
                                        "tol": tol,
                                        "l1_ratio": l1_ratio
                                    })
                            else:
                                combinations.append({
                                    "penalty": penalty,
                                    "solver": "saga",
                                    "C": C,
                                    "class_weight": class_weight,
                                    "max_iter": max_iter,
                                    "tol": tol
                                })
        
        # For liblinear solver (supports l1, l2)
        for penalty in ["l1", "l2"]:
            for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
                for class_weight in [None, "balanced"]:
                    for max_iter in [1000, 2000, 3000]:
                        for tol in [1e-6, 1e-5, 1e-4]:
                            combinations.append({
                                "penalty": penalty,
                                "solver": "liblinear",
                                "C": C,
                                "class_weight": class_weight,
                                "max_iter": max_iter,
                                "tol": tol
                            })
        
        # For newton-cg, lbfgs (support l2)
        for solver in ["newton-cg", "lbfgs"]:
            for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
                for class_weight in [None, "balanced"]:
                    for max_iter in [1000, 2000, 3000]:
                        for tol in [1e-6, 1e-5, 1e-4]:
                            combinations.append({
                                "penalty": "l2",
                                "solver": solver,
                                "C": C,
                                "class_weight": class_weight,
                                "max_iter": max_iter,
                                "tol": tol
                            })
        
        # Randomly sample from these combinations
        if len(combinations) > 100:
            import random
            random.seed(42)
            combinations = random.sample(combinations, 100)
        
        return combinations
    
    # For other models, return the existing dictionary
    return PARAM_DISTRIBUTIONS.get(model_name, {})

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
    if model_name == "logistic_regression":
        # For logistic regression, we'll directly provide parameter combinations
        param_dicts = get_param_combinations(model_name)
        param_dist = param_dicts
    else:
        param_dist = PARAM_DISTRIBUTIONS[model_name]
    
    # Set up cross-validation with stratification to handle potential class imbalance
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Define multiple scoring metrics for comprehensive evaluation
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    # Use multiprocessing for faster tuning
    n_jobs = min(os.cpu_count() - 1, 4)  # Leave at least 1 CPU free
    if n_jobs < 1:
        n_jobs = 1
    
    # Evaluate base model before tuning
    logger.info(f"Evaluating base model before tuning...")
    base_model.fit(X_train, y_train)
    before_metrics = evaluate_model(base_model, X_test, y_test)
    logger.info(f"Performance before tuning: F1 = {before_metrics['f1']:.4f}, Accuracy = {before_metrics['accuracy']:.4f}")
    
    # Start timing
    start_time = time.time()
    
    # Perform randomized search with multiple scoring metrics
    logger.info(f"Starting hyperparameter tuning for {model_name} with {n_iter} iterations...")
    
    # For logistic regression, use a different approach
    if model_name == "logistic_regression":
        # Limit the number of iterations to the number of parameter combinations or n_iter, whichever is smaller
        n_iter_actual = min(n_iter, len(param_dicts))
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dicts,
            n_iter=n_iter_actual,
            cv=cv_strategy,
            scoring=scoring_metrics if scoring == 'all' else scoring,
            refit=scoring,  # Refit on the best parameters for the specified scoring metric
            n_jobs=n_jobs,
            verbose=2,
            random_state=42,
            return_train_score=True,
            error_score='raise'  # Raises error for debugging
        )
    else:
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv_strategy,
            scoring=scoring_metrics if scoring == 'all' else scoring,
            refit=scoring,  # Refit on the best parameters for the specified scoring metric
            n_jobs=n_jobs,
            verbose=2,
            random_state=42,
            return_train_score=True,
            error_score='raise'  # Raises error for debugging
        )
    
    # Train the model
    try:
        random_search.fit(X_train, y_train)
        
        # Capture all scoring results if multiple metrics were used
        cv_results = {}
        if scoring == 'all':
            for metric in scoring_metrics.keys():
                cv_results[f"mean_test_{metric}"] = float(random_search.cv_results_[f"mean_test_{metric}"][random_search.best_index_])
                cv_results[f"std_test_{metric}"] = float(random_search.cv_results_[f"std_test_{metric}"][random_search.best_index_])
        else:
            cv_results["mean_test_score"] = float(random_search.cv_results_["mean_test_score"][random_search.best_index_])
            cv_results["std_test_score"] = float(random_search.cv_results_["std_test_score"][random_search.best_index_])
            cv_results["mean_train_score"] = float(random_search.cv_results_["mean_train_score"][random_search.best_index_])
            cv_results["std_train_score"] = float(random_search.cv_results_["std_train_score"][random_search.best_index_])
    
    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {e}")
        return None, None, None
    
    # End timing
    tuning_time = time.time() - start_time
    
    # Get the best model and parameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    # Evaluate tuned model
    after_metrics = evaluate_model(best_model, X_test, y_test)
    logger.info(f"Performance after tuning: F1 = {after_metrics['f1']:.4f}, Accuracy = {after_metrics['accuracy']:.4f}")
    
    # Calculate improvement for each metric
    improvements = {}
    improvement_percents = {}
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        abs_improvement = after_metrics[metric] - before_metrics[metric]
        improvements[metric] = float(abs_improvement)
        
        if before_metrics[metric] > 0:
            pct_improvement = (abs_improvement / before_metrics[metric]) * 100
        else:
            pct_improvement = float('inf')
        
        improvement_percents[metric] = float(pct_improvement)
        logger.info(f"{metric.upper()} improvement: {abs_improvement:.4f} ({pct_improvement:.2f}%)")
    
    # Check for overfitting
    train_score = random_search.cv_results_["mean_train_score"][random_search.best_index_]
    test_score = random_search.cv_results_["mean_test_score"][random_search.best_index_]
    
    if train_score > test_score * 1.1:  # 10% threshold
        logger.warning("Possible overfitting detected: Train score significantly higher than test score")
    
    # Compile tuning results
    tuning_results = {
        "model_name": model_name,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "best_params": best_params,
        "cv_results": cv_results,
        "improvement": {
            "absolute": improvements,
            "percent": improvement_percents
        },
        "tuning_time": float(tuning_time),
        "n_iter": n_iter,
        "cv": cv,
        "scoring": scoring,
        "timestamp": datetime.now().isoformat(),
        "overfitting_check": {
            "train_score": float(train_score),
            "test_score": float(test_score),
            "difference": float(train_score - test_score)
        }
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
        
        # Convert numpy types to native Python types to ensure JSON serializability
        serializable_results = convert_to_serializable(tuning_results)
        
        # Save tuning results
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Saved tuning results to {results_file}")
        
        return model_file, results_file
    
    except Exception as e:
        logger.error(f"Error saving tuned model or results: {e}")
        return None, None

def convert_to_serializable(obj):
    """
    Convert an object to a JSON serializable format.
    
    Args:
        obj: The object to convert
    
    Returns:
        A JSON serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (datetime, np.datetime64)):
        return obj.isoformat()
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    else:
        return str(obj)  # Fallback to string representation

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
        choices=["accuracy", "precision", "recall", "f1", "roc_auc", "all"],
        default="f1",
        help="Scoring metric for hyperparameter tuning (default: f1)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Custom directory to save tuning results (defaults to models/tuning_results)"
    )
    parser.add_argument(
        "--optimize_all",
        action="store_true",
        help="Run a more comprehensive tuning process with higher iterations"
    )
    args = parser.parse_args()
    
    # If optimize_all is set, increase iterations
    if args.optimize_all:
        args.n_iter = 100  # Double the default iterations
        logger.info(f"Comprehensive optimization requested: increased to {args.n_iter} iterations")
    
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
    
    # Set custom output directory if provided
    if args.output_dir:
        global TUNING_DIR
        TUNING_DIR = Path(args.output_dir)
        TUNING_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using custom output directory: {TUNING_DIR}")
    
    logger.info(f"Starting hyperparameter tuning for tasks: {tasks}, models: {models}")
    logger.info(f"Using {args.n_iter} iterations, {args.cv} CV folds, and {args.scoring} scoring")
    
    # Track overall results
    results_summary = {
        "tasks": {},
        "total_tuning_time": 0,
        "started_at": datetime.now().isoformat(),
    }
    
    total_start_time = time.time()
    
    # Tune models for each task
    for task in tasks:
        results_summary["tasks"][task] = {"models": {}}
        
        for model_name in models:
            logger.info(f"Tuning {model_name} for {task}")
            model_start_time = time.time()
            
            model_path, results_path = tune_model_for_task(
                model_name, task, args.n_iter, args.cv, args.scoring
            )
            
            model_duration = time.time() - model_start_time
            
            if model_path is not None:
                logger.info(f"Successfully tuned {model_name} for {task}")
                logger.info(f"Model saved to: {model_path}")
                logger.info(f"Results saved to: {results_path}")
                logger.info(f"Tuning duration: {model_duration:.2f} seconds")
                
                # Try to load tuning results to extract metrics
                try:
                    with open(results_path, 'r') as f:
                        tuning_results = json.load(f)
                    
                    # Save summary metrics
                    results_summary["tasks"][task]["models"][model_name] = {
                        "before": {
                            "f1": tuning_results["before_metrics"]["f1"],
                            "accuracy": tuning_results["before_metrics"]["accuracy"]
                        },
                        "after": {
                            "f1": tuning_results["after_metrics"]["f1"],
                            "accuracy": tuning_results["after_metrics"]["accuracy"]
                        },
                        "improvement": {
                            "f1_percent": tuning_results["improvement"]["percent"]["f1"],
                            "accuracy_percent": tuning_results["improvement"]["percent"]["accuracy"]
                        },
                        "duration": model_duration
                    }
                except Exception as e:
                    logger.error(f"Error extracting metrics for summary: {e}")
                    results_summary["tasks"][task]["models"][model_name] = {
                        "status": "completed",
                        "duration": model_duration
                    }
            else:
                logger.error(f"Failed to tune {model_name} for {task}")
                results_summary["tasks"][task]["models"][model_name] = {
                    "status": "failed",
                    "duration": model_duration
                }
    
    # Calculate total tuning time
    total_duration = time.time() - total_start_time
    results_summary["total_tuning_time"] = total_duration
    results_summary["completed_at"] = datetime.now().isoformat()
    
    # Save summary
    summary_path = TUNING_DIR / f"tuning_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        logger.info(f"Saved tuning summary to: {summary_path}")
    except Exception as e:
        logger.error(f"Error saving tuning summary: {e}")
    
    logger.info(f"Hyperparameter tuning completed in {total_duration:.2f} seconds")

if __name__ == "__main__":
    main() 