#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Feature importance analysis for sleep efficiency and mood prediction models.
# Provides data-driven recommendations for model refinement.

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
    # Loads feature data for a specific task from joblib file
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
    # Loads all models for a specific task and returns dictionary of models and metadata
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

def analyze_feature_importance(models_dict, X_train, y_train, feature_names):
    # Analyzes and visualizes feature importance across different models
    importance_results = {}
    
    # Create figure for plotting
    plt.figure(figsize=(15, 10))
    
    model_count = 0
    for model_name, model_info in models_dict.items():
        model = model_info["model"]
        metadata = model_info["metadata"]
        
        # Get expected features from metadata
        expected_features = None
        if "selected_features" in metadata:
            expected_features = metadata["selected_features"]
        elif "features" in metadata:
            expected_features = metadata["features"]
        
        # If no feature information in metadata, use the provided feature_names
        if expected_features is None:
            logger.info(f"No feature information found in metadata for {model_name}, using provided feature_names")
            expected_features = feature_names
        
        try:
            # Get feature importance based on model type
            if hasattr(model, 'feature_importances_'):
                # Model has built-in feature importance
                importance = model.feature_importances_
                importance_type = "built-in"
                
                # For XGBoost models, make sure the importances match the number of features
                if len(importance) != len(expected_features):
                    if len(importance) > len(expected_features):
                        importance = importance[:len(expected_features)]
                        logger.warning(f"Truncated feature importance for {model_name}")
                    else:
                        # Not enough importances, need to pad
                        logger.warning(f"Feature importance length mismatch for {model_name}, skipping")
                        continue
            else:
                # Use permutation importance with basic dataset
                # This may not be accurate, but it's better than nothing
                logger.info(f"Using permutation importance for {model_name}")
                
                # Make sure X_train and y_train are numpy arrays
                if not isinstance(X_train, np.ndarray):
                    X_train_array = X_train.values
                else:
                    X_train_array = X_train
                    
                if not isinstance(y_train, np.ndarray):
                    y_train_array = y_train.values
                else:
                    y_train_array = y_train
                
                # Use permutation importance with just a few repeats
                try:
                    perm_importance = permutation_importance(model, X_train_array, y_train_array, n_repeats=2, random_state=42)
                    importance = perm_importance.importances_mean
                    importance_type = "permutation"
                except Exception as e:
                    logger.error(f"Error calculating permutation importance for {model_name}: {e}")
                    continue
            
            # Create a DataFrame for easier handling
            if len(importance) != len(expected_features):
                logger.warning(f"Feature importance length ({len(importance)}) doesn't match expected features length ({len(expected_features)}) for {model_name}")
                # Use the shorter length
                min_length = min(len(importance), len(expected_features))
                importance_df = pd.DataFrame({
                    'feature': expected_features[:min_length],
                    'importance': importance[:min_length]
                })
            else:
                importance_df = pd.DataFrame({
                    'feature': expected_features,
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
            model_count += 1
            plt.subplot(3, 4, model_count)
            if len(importance_df) > 10:
                plt.barh(importance_df['feature'][:10], importance_df['importance'][:10])
            else:
                plt.barh(importance_df['feature'], importance_df['importance'])
            plt.xlabel('Importance')
            plt.title(f'{model_name}')
            
            logger.info(f"Analyzed feature importance for {model_name}")
        except Exception as e:
            logger.error(f"Error analyzing feature importance for {model_name}: {e}")
    
    # Save the plot if we have any results
    if importance_results:
        plt.tight_layout()
        plt.savefig(ANALYSIS_DIR / "feature_importance.png")
        logger.info(f"Saved feature importance plot to {ANALYSIS_DIR / 'feature_importance.png'}")
    else:
        logger.warning("No feature importance results to plot")
    
    return importance_results

def refine_model(model_name, X_train, y_train, X_test, y_test, feature_importance=None):
    # Refines model using feature selection and hyperparameter tuning
    logger.info(f"Refining {model_name} model")
    
    # Feature selection based on importance
    if feature_importance is not None:
        # Select top features (top 75%)
        top_features = feature_importance.nlargest(int(len(feature_importance) * 0.75), 'importance')
        selected_features = top_features['feature'].tolist()
        
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        
        # Convert indices to column names if needed
        if isinstance(X_train, np.ndarray):
            # For NumPy arrays, we need the indices
            selected_indices = [list(feature_importance['feature']).index(feat) for feat in selected_features]
            X_train_selected = X_train[:, selected_indices]
            X_test_selected = X_test[:, selected_indices]
        else:
            # For pandas DataFrames
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]
    else:
        # Use all features if no importance information is available
        logger.info("No feature importance provided, using all features")
        if isinstance(X_train, np.ndarray):
            X_train_selected = X_train
            X_test_selected = X_test
            selected_features = list(range(X_train.shape[1]))  # Use indices for NumPy arrays
        else:
            X_train_selected = X_train
            X_test_selected = X_test
            selected_features = X_train.columns.tolist()
    
    # Initialize model based on model name
    if model_name.startswith("xgboost"):
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
    
    elif model_name.startswith("random_forest"):
        # Define parameter grid for Random Forest
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Initialize model
        model = RandomForestClassifier(random_state=42)
    
    else:
        # Skip model refinement for unknown models
        logger.warning(f"Model type {model_name} not supported for refinement")
        return None, None
    
    # Perform basic grid search with a small sample of parameters
    limited_param_grid = {}
    for param, values in param_grid.items():
        if len(values) > 2:
            limited_param_grid[param] = [values[0], values[-1]]
        else:
            limited_param_grid[param] = values
    
    try:
        # Create a smaller grid search to save time
        logger.info(f"Running grid search for {model_name} with {len(limited_param_grid)} parameters")
        grid_search = GridSearchCV(model, limited_param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train_selected, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        logger.info(f"Best parameters: {best_params}")
        
        # Evaluate refined model
        y_pred = best_model.predict(X_test_selected)
        if hasattr(best_model, "predict_proba"):
            y_prob = best_model.predict_proba(X_test_selected)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            roc_auc = 0.0
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred)),
            'roc_auc': float(roc_auc)
        }
        
        logger.info(f"Refined model metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Create metadata
        metadata = {
            'best_params': best_params,
            'metrics': metrics,
            'selected_features': selected_features,
            'refinement_date': pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        }
        
        return best_model, metadata
    
    except Exception as e:
        logger.error(f"Error refining model {model_name}: {e}")
        return None, None

def analyze_feature_correlations(X_train, feature_names):
    # Analyzes correlations between features to identify redundancies
    logger.info("Analyzing feature correlations")
    
    # Convert to DataFrame if X_train is a numpy array
    if isinstance(X_train, np.ndarray):
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
    else:
        X_train_df = X_train
    
    # Calculate correlation matrix
    corr_matrix = X_train_df.corr()
    
    # Find highly correlated features
    high_correlations = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:  # Threshold for high correlation
                high_correlations.append({
                    'feature1': feature_names[i],
                    'feature2': feature_names[j],
                    'correlation': float(corr_matrix.iloc[i, j])
                })
    
    return {
        'correlation_matrix': corr_matrix,
        'high_correlations': high_correlations
    }

def main():
    # Main function to analyze feature importance and refine models
    logger.info("Starting sleep feature analysis")
    
    # Load features for mood prediction
    features = load_features("mood_prediction")
    if features is None:
        logger.error("Failed to load features")
        return 1
    
    # Extract features and labels
    X_train = features['X_train']
    X_test = features['X_test']
    y_train = features['y_train']
    y_test = features['y_test']
    feature_names = features['feature_names']
    
    logger.info(f"Loaded {len(feature_names)} features")
    
    # Load models
    models_dict = load_models("mood_prediction")
    if not models_dict:
        logger.error("No models found")
        return 1
    
    logger.info(f"Loaded {len(models_dict)} models")
    
    # Analyze feature importance
    importance_results = analyze_feature_importance(models_dict, X_train, y_train, feature_names)
    
    # Analyze feature correlations
    correlation_results = analyze_feature_correlations(X_train, feature_names)
    
    # Print high correlations
    if correlation_results['high_correlations']:
        logger.info("\nHighly correlated features:")
        for corr in correlation_results['high_correlations']:
            logger.info(f"  {corr['feature1']} and {corr['feature2']}: {corr['correlation']:.4f}")
    
    # Refine models
    refined_models = {}
    for model_name in models_dict.keys():
        # Get feature importance for this model
        if model_name in importance_results:
            feature_importance = importance_results[model_name]['importance']
        else:
            feature_importance = None
        
        # Refine model
        refined_model, metadata = refine_model(
            model_name, X_train, y_train, X_test, y_test, feature_importance
        )
        
        if refined_model is not None:
            refined_models[model_name] = {
                'model': refined_model,
                'metadata': metadata
            }
            
            # Save refined model
            model_path = MODELS_DIR / "mood_prediction" / f"{model_name}_refined.joblib"
            metadata_path = MODELS_DIR / "mood_prediction" / f"{model_name}_refined_metadata.joblib"
            
            joblib.dump(refined_model, model_path)
            joblib.dump(metadata, metadata_path)
            
            logger.info(f"Saved refined {model_name} model to {model_path}")
    
    # Compare original and refined models
    logger.info("\nModel comparison (original vs refined):")
    for model_name, refined_info in refined_models.items():
        original_metrics = models_dict[model_name]['metadata'].get('metrics', {})
        refined_metrics = refined_info['metadata'].get('metrics', {})
        
        logger.info(f"\n{model_name}:")
        for metric in ['accuracy', 'f1', 'roc_auc']:
            if metric in original_metrics and metric in refined_metrics:
                improvement = refined_metrics[metric] - original_metrics[metric]
                logger.info(f"  {metric}: {original_metrics[metric]:.4f} → {refined_metrics[metric]:.4f} " +
                          f"({'↑' if improvement > 0 else '↓'}{abs(improvement):.4f})")
    
    logger.info("\nAnalysis completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 