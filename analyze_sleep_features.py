#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script analyzes feature importance in sleep efficiency and mood prediction models
and provides recommendations for model refinement.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
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
MODELS_DIR = ROOT_DIR / "models"
ANALYSIS_DIR = ROOT_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

def load_features(task="mood_prediction"):
    """
    Load features for a specific task.
    
    Args:
        task (str): Task name ('mood_prediction')
    
    Returns:
        dict: Dictionary containing features and labels
    """
    features_file = FEATURES_DIR / f"{task}_features.joblib"
    
    if not features_file.exists():
        logger.error(f"Features file not found: {features_file}")
        return None
    
    try:
        features = joblib.load(features_file)
        logger.info(f"Loaded features for {task} from {features_file}")
        return features
    
    except Exception as e:
        logger.error(f"Error loading features: {e}")
        return None

def load_models(task="mood_prediction"):
    """
    Load all models for a specific task.
    
    Args:
        task (str): Task name ('mood_prediction')
    
    Returns:
        dict: Dictionary of models and their metadata
    """
    models_dict = {}
    model_dir = MODELS_DIR / task
    
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        return models_dict
    
    # Look for model files
    model_files = list(model_dir.glob("*.joblib"))
    
    for model_file in model_files:
        if "_metadata" in model_file.name:
            continue
            
        model_name = model_file.stem
        metadata_file = model_dir / f"{model_name}_metadata.joblib"
        
        if not metadata_file.exists():
            logger.warning(f"Metadata file not found for {model_name}")
            continue
            
        try:
            model = joblib.load(model_file)
            metadata = joblib.load(metadata_file)
            
            models_dict[model_name] = {
                "model": model,
                "metadata": metadata,
                "file_path": str(model_file)
            }
            
            logger.info(f"Loaded {model_name} model")
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
    
    return models_dict

def analyze_feature_importance(models_dict, X_train, feature_names):
    """
    Analyze feature importance across different models.
    
    Args:
        models_dict (dict): Dictionary of models
        X_train (pd.DataFrame): Training features
        feature_names (list): List of feature names
    
    Returns:
        dict: Dictionary of feature importance results
    """
    importance_results = {}
    
    # Create figure for plotting
    plt.figure(figsize=(12, 8))
    
    for model_name, model_info in models_dict.items():
        model = model_info["model"]
        
        # Get feature importance based on model type
        if model_name == "xgboost":
            # XGBoost has built-in feature importance
            try:
                importance = model.feature_importances_
                importance_type = "built-in"
            except:
                # Use permutation importance as fallback
                perm_importance = permutation_importance(model, X_train, feature_names=feature_names, n_repeats=10, random_state=42)
                importance = perm_importance.importances_mean
                importance_type = "permutation"
                
        elif model_name == "random_forest":
            # Random Forest has built-in feature importance
            try:
                importance = model.feature_importances_
                importance_type = "built-in"
            except:
                # Use permutation importance as fallback
                perm_importance = permutation_importance(model, X_train, feature_names=feature_names, n_repeats=10, random_state=42)
                importance = perm_importance.importances_mean
                importance_type = "permutation"
                
        else:
            # For other models, use permutation importance
            perm_importance = permutation_importance(model, X_train, feature_names=feature_names, n_repeats=10, random_state=42)
            importance = perm_importance.importances_mean
            importance_type = "permutation"
        
        # Create a DataFrame for easier handling
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Store results
        importance_results[model_name] = {
            'type': importance_type,
            'importance': importance_df
        }
        
        # Plot feature importance
        plt.subplot(2, 2, list(models_dict.keys()).index(model_name) + 1)
        plt.barh(importance_df['feature'][:10], importance_df['importance'][:10])
        plt.xlabel('Importance')
        plt.title(f'{model_name} Feature Importance')
        plt.tight_layout()
    
    # Save the plot
    plt.savefig(ANALYSIS_DIR / "feature_importance.png")
    logger.info(f"Saved feature importance plot to {ANALYSIS_DIR / 'feature_importance.png'}")
    
    return importance_results

def refine_model(model_name, X_train, y_train, X_test, y_test, feature_importance=None):
    """
    Refine a model using feature selection and hyperparameter tuning.
    
    Args:
        model_name (str): Name of the model to refine
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        feature_importance (pd.DataFrame): Feature importance dataframe
    
    Returns:
        tuple: (refined_model, performance_metrics)
    """
    logger.info(f"Refining {model_name} model")
    
    # Feature selection based on importance
    if feature_importance is not None:
        # Select top features (top 75%)
        top_features = feature_importance.nlargest(int(len(feature_importance) * 0.75), 'importance')
        selected_features = top_features['feature'].tolist()
        
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        
        # Select features from data
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
    else:
        # Use all features if no importance information is available
        X_train_selected = X_train
        X_test_selected = X_test
        selected_features = X_train.columns.tolist()
    
    # Initialize model based on model name
    if model_name == "xgboost":
        # Define parameter grid for XGBoost
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5, 6],
            'n_estimators': [50, 100, 200],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2],
            'min_child_weight': [1, 3, 5]
        }
        
        # Initialize model
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        
    elif model_name == "random_forest":
        # Define parameter grid for Random Forest
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        # Initialize model
        model = RandomForestClassifier(random_state=42)
        
    else:
        logger.error(f"Unsupported model: {model_name}")
        return None, None
    
    # Perform grid search
    logger.info(f"Performing grid search for {model_name}")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_selected, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    logger.info(f"Best parameters: {best_params}")
    
    # Evaluate model
    y_pred = best_model.predict(X_test_selected)
    y_prob = best_model.predict_proba(X_test_selected)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    logger.info(f"Performance metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save refined model
    refined_model_path = MODELS_DIR / "mood_prediction" / f"{model_name}_refined.joblib"
    refined_metadata_path = MODELS_DIR / "mood_prediction" / f"{model_name}_refined_metadata.joblib"
    
    # Prepare metadata
    metadata = {
        'model_name': f"{model_name}_refined",
        'task': 'mood_prediction',
        'metrics': metrics,
        'params': best_params,
        'selected_features': selected_features,
        'description': f'Refined {model_name} model with feature selection and hyperparameter tuning'
    }
    
    # Save model and metadata
    joblib.dump(best_model, refined_model_path)
    joblib.dump(metadata, refined_metadata_path)
    
    logger.info(f"Saved refined model to {refined_model_path}")
    logger.info(f"Saved refined metadata to {refined_metadata_path}")
    
    return best_model, metrics

def analyze_feature_correlations(X_train, feature_names):
    """
    Analyze correlations between features.
    
    Args:
        X_train (pd.DataFrame): Training features
        feature_names (list): List of feature names
    """
    # Create DataFrame with named columns
    X_df = pd.DataFrame(X_train, columns=feature_names)
    
    # Calculate correlation matrix
    corr_matrix = X_df.corr()
    
    # Create a heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(ANALYSIS_DIR / "feature_correlations.png")
    logger.info(f"Saved feature correlation matrix to {ANALYSIS_DIR / 'feature_correlations.png'}")
    
    # Find highly correlated features
    high_corr_features = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_features.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    if high_corr_features:
        logger.info("Highly correlated features (|r| > 0.8):")
        for feat1, feat2, corr in high_corr_features:
            logger.info(f"  {feat1} - {feat2}: {corr:.4f}")
    else:
        logger.info("No highly correlated features found (|r| > 0.8)")
    
    return corr_matrix, high_corr_features

def main():
    """Main function to analyze features and refine models."""
    # Load features
    features = load_features()
    
    if features is None:
        return 1
    
    # Extract training and test data
    X_train = features["X_train"]
    y_train = features["y_train"]
    X_test = features["X_test"]
    y_test = features["y_test"]
    feature_names = features["feature_names"]
    
    # Load models
    models_dict = load_models()
    
    if not models_dict:
        logger.error("No models found")
        return 1
    
    # Analyze feature importance
    importance_results = analyze_feature_importance(models_dict, X_train, feature_names)
    
    # Analyze feature correlations
    corr_matrix, high_corr_features = analyze_feature_correlations(X_train, feature_names)
    
    # Refine selected models
    for model_name in ["xgboost", "random_forest"]:
        if model_name in models_dict:
            # Get feature importance dataframe
            importance_df = importance_results[model_name]['importance']
            
            # Refine model
            refined_model, metrics = refine_model(model_name, X_train, y_train, X_test, y_test, importance_df)
            
            if refined_model is None:
                logger.error(f"Failed to refine {model_name} model")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 