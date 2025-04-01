#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implements and evaluates baseline classification models for mood prediction:
- Logistic Regression: Baseline linear classifier
- SVM: For capturing non-linear patterns using RBF kernel
- Random Forest: Non-parametric approach for handling EEG noise
- XGBoost: For comparison with the project's primary model

This script trains all models with hyperparameter tuning and evaluates them
on the same test set for fair comparison.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
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

# Create directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_features():
    """
    Load mood prediction features from the project's existing feature extraction.
    
    Returns:
        dict: Dictionary containing X_train, X_test, y_train, y_test, feature_names
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

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    
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
    logger.info(f"Model evaluation metrics:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    
    # Print classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    return metrics

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a Logistic Regression model with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
    
    Returns:
        tuple: (model, metrics, best_params)
    """
    logger.info("Training Logistic Regression model with hyperparameter tuning")
    
    # Define parameter grid
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],  # liblinear supports both l1 and l2
        "class_weight": [None, "balanced"]
    }
    
    # Initialize model
    lr = LogisticRegression(
        random_state=42,
        max_iter=2000
    )
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=lr,
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
    
    logger.info(f"Best parameters for Logistic Regression: {best_params}")
    logger.info(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    metrics = evaluate_model(best_model, X_test, y_test)
    
    # Save model
    model_file = MODELS_DIR / "logistic_regression.joblib"
    metadata_file = MODELS_DIR / "logistic_regression_metadata.joblib"
    
    metadata = {
        "model_name": "logistic_regression",
        "metrics": metrics,
        "best_params": best_params,
        "cv_score": float(grid_search.best_score_)
    }
    
    joblib.dump(best_model, model_file)
    joblib.dump(metadata, metadata_file)
    logger.info(f"Saved model to {model_file}")
    
    return best_model, metrics, best_params

def train_svm(X_train, y_train, X_test, y_test):
    """
    Train and evaluate an SVM model with RBF kernel and hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
    
    Returns:
        tuple: (model, metrics, best_params)
    """
    logger.info("Training SVM model with RBF kernel and hyperparameter tuning")
    
    # Define parameter grid focusing on RBF kernel
    param_grid = {
        "C": [0.1, 1.0, 10.0, 100.0],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
        "class_weight": [None, "balanced"]
    }
    
    # Initialize model with RBF kernel
    svm = SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=svm,
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
    
    logger.info(f"Best parameters for SVM with RBF kernel: {best_params}")
    logger.info(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    metrics = evaluate_model(best_model, X_test, y_test)
    
    # Save model
    model_file = MODELS_DIR / "svm_rbf.joblib"
    metadata_file = MODELS_DIR / "svm_rbf_metadata.joblib"
    
    metadata = {
        "model_name": "svm_rbf",
        "metrics": metrics,
        "best_params": best_params,
        "cv_score": float(grid_search.best_score_)
    }
    
    joblib.dump(best_model, model_file)
    joblib.dump(metadata, metadata_file)
    logger.info(f"Saved model to {model_file}")
    
    return best_model, metrics, best_params

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a Random Forest model with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
    
    Returns:
        tuple: (model, metrics, best_params)
    """
    logger.info("Training Random Forest model with hyperparameter tuning")
    
    # Define parameter grid
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
        "class_weight": [None, "balanced", "balanced_subsample"]
    }
    
    # Initialize model
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=rf,
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
    
    logger.info(f"Best parameters for Random Forest: {best_params}")
    logger.info(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    metrics = evaluate_model(best_model, X_test, y_test)
    
    # Save model
    model_file = MODELS_DIR / "random_forest.joblib"
    metadata_file = MODELS_DIR / "random_forest_metadata.joblib"
    
    metadata = {
        "model_name": "random_forest",
        "metrics": metrics,
        "best_params": best_params,
        "cv_score": float(grid_search.best_score_)
    }
    
    joblib.dump(best_model, model_file)
    joblib.dump(metadata, metadata_file)
    logger.info(f"Saved model to {model_file}")
    
    return best_model, metrics, best_params

def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train and evaluate an XGBoost model with hyperparameter tuning for comparison.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
    
    Returns:
        tuple: (model, metrics, best_params)
    """
    logger.info("Training XGBoost model with hyperparameter tuning")
    
    # Define parameter grid
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2],
        "min_child_weight": [1, 3, 5],
        "scale_pos_weight": [1, 3, 5, 7]
    }
    
    # Initialize model
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=xgb_model,
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
    
    logger.info(f"Best parameters for XGBoost: {best_params}")
    logger.info(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    metrics = evaluate_model(best_model, X_test, y_test)
    
    # Save model
    model_file = MODELS_DIR / "xgboost.joblib"
    metadata_file = MODELS_DIR / "xgboost_metadata.joblib"
    
    metadata = {
        "model_name": "xgboost",
        "metrics": metrics,
        "best_params": best_params,
        "cv_score": float(grid_search.best_score_)
    }
    
    joblib.dump(best_model, model_file)
    joblib.dump(metadata, metadata_file)
    logger.info(f"Saved model to {model_file}")
    
    return best_model, metrics, best_params

def visualize_model_comparison(results):
    """
    Create visualizations to compare model performance.
    
    Args:
        results: Dictionary of model results
    """
    # Extract metrics for comparison
    model_names = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    
    # Create DataFrame for plotting
    data = []
    for model_name in model_names:
        model_metrics = results[model_name]["metrics"]
        
        for metric in metrics:
            data.append({
                "Model": model_name,
                "Metric": metric,
                "Value": model_metrics[metric]
            })
    
    df = pd.DataFrame(data)
    
    # Plot metrics comparison
    plt.figure(figsize=(14, 8))
    
    # Create colorful bar plot
    ax = sns.barplot(x="Model", y="Value", hue="Metric", data=df, palette="viridis")
    
    # Add labels and title
    plt.title("Model Performance Comparison", fontsize=16)
    plt.ylabel("Score", fontsize=14)
    plt.xlabel("Model", fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Metric", title_fontsize=12, fontsize=10, loc="best")
    
    # Add value labels on bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}",
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'bottom',
                   xytext = (0, 5),
                   textcoords = 'offset points',
                   fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    metrics_plot_path = RESULTS_DIR / "model_metrics_comparison.png"
    plt.savefig(metrics_plot_path, dpi=300)
    logger.info(f"Saved metrics comparison plot to {metrics_plot_path}")
    
    # Create confusion matrix plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, model_name in enumerate(model_names):
        cm = np.array(results[model_name]["metrics"]["confusion_matrix"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axes[i], values_format="d", colorbar=False)
        disp.ax_.set_title(f"{model_name}", fontsize=14)
        disp.ax_.set_xlabel("Predicted label", fontsize=12)
        disp.ax_.set_ylabel("True label", fontsize=12)
    
    plt.tight_layout()
    
    # Save the confusion matrix plot
    cm_plot_path = RESULTS_DIR / "model_confusion_matrices.png"
    plt.savefig(cm_plot_path, dpi=300)
    logger.info(f"Saved confusion matrices plot to {cm_plot_path}")
    
    # Save results as JSON
    results_file = RESULTS_DIR / "model_comparison_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_file}")

def main():
    """
    Main function to train and evaluate all models.
    """
    logger.info("Starting baseline model training and evaluation")
    
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
    
    # Train and evaluate all models
    results = {}
    
    # 1. Logistic Regression
    lr_model, lr_metrics, lr_params = train_logistic_regression(X_train, y_train, X_test, y_test)
    results["logistic_regression"] = {
        "model": lr_model,
        "metrics": lr_metrics,
        "best_params": lr_params
    }
    
    # 2. SVM with RBF kernel
    svm_model, svm_metrics, svm_params = train_svm(X_train, y_train, X_test, y_test)
    results["svm_rbf"] = {
        "model": svm_model,
        "metrics": svm_metrics,
        "best_params": svm_params
    }
    
    # 3. Random Forest
    rf_model, rf_metrics, rf_params = train_random_forest(X_train, y_train, X_test, y_test)
    results["random_forest"] = {
        "model": rf_model,
        "metrics": rf_metrics,
        "best_params": rf_params
    }
    
    # 4. XGBoost for comparison
    xgb_model, xgb_metrics, xgb_params = train_xgboost(X_train, y_train, X_test, y_test)
    results["xgboost"] = {
        "model": xgb_model,
        "metrics": xgb_metrics,
        "best_params": xgb_params
    }
    
    # Create visualizations and save results
    visualize_model_comparison(results)
    
    # Identify best model
    best_model = max(results.keys(), key=lambda k: results[k]["metrics"]["f1"])
    best_f1 = results[best_model]["metrics"]["f1"]
    
    logger.info("\n" + "="*50)
    logger.info(f"Best model: {best_model} with F1 score: {best_f1:.4f}")
    logger.info("="*50 + "\n")
    
    logger.info("Model training and evaluation completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 