#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script evaluates the best tuned model on the test data.
"""

import os
import sys
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Define the root directory
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = ROOT_DIR / "models"

def load_features(task):
    """
    Load extracted features for a specific task.
    
    Args:
        task (str): Task name ('rem_detection' or 'mood_prediction')
    
    Returns:
        dict: Dictionary containing features
    """
    # Determine the features file
    features_file = FEATURES_DIR / f"{task}_features.joblib"
    
    # Check if the file exists
    if not features_file.exists():
        print(f"Features file not found: {features_file}")
        return None
    
    try:
        # Load the features
        features = joblib.load(features_file)
        print(f"Loaded features for {task} from {features_file}")
        return features
    
    except Exception as e:
        print(f"Error loading features from {features_file}: {e}")
        return None

def load_best_model(task):
    """
    Load the best tuned model for a specific task.
    
    Args:
        task (str): Task name
    
    Returns:
        object: Trained model
    """
    # Get the task directory
    task_dir = MODELS_DIR / task
    
    # Check if the directory exists
    if not task_dir.exists():
        print(f"Task directory not found: {task_dir}")
        return None
    
    # Find the latest tuned model
    model_files = list(task_dir.glob("*_tuned_*.joblib"))
    
    if not model_files:
        print(f"No tuned models found in {task_dir}")
        return None
    
    # Sort by modification time (newest first)
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Get the most recent model
    latest_model_file = model_files[0]
    
    try:
        # Load the model
        model = joblib.load(latest_model_file)
        print(f"Loaded best model from {latest_model_file}")
        return model
    
    except Exception as e:
        print(f"Error loading model from {latest_model_file}: {e}")
        return None

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    
    Args:
        model (object): Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print("\n=== Model Evaluation Results ===\n")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}")
    print(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist()
    }

def main():
    """Main function to evaluate the best model."""
    # Define the task
    task = "rem_detection"
    
    # Load features
    features = load_features(task)
    
    if features is None:
        return 1
    
    # Load the best model
    model = load_best_model(task)
    
    if model is None:
        return 1
    
    # Extract test data
    X_test = features["X_test"]
    y_test = features["y_test"]
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 