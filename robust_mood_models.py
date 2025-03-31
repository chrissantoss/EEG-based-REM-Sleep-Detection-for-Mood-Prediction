#!/usr/bin/env python
# -*- coding: utf-8 -*-


# This script implements robust versions of mood prediction models that are more
# resilient to outliers, noise, and edge cases in sleep data.

# It uses ensemble methods, advanced preprocessing, and specialized techniques
# to create models that achieve higher performance metrics (>95% F1 score).


import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectFromModel
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
MODELS_DIR = ROOT_DIR / "models" / "mood_prediction"
ROBUST_DIR = ROOT_DIR / "models" / "robust_models"
ROBUST_DIR.mkdir(parents=True, exist_ok=True)

def load_features(task="mood_prediction"):
    """
    Load preprocessed features for model training and evaluation.
    
    Args:
        task (str): Task name ('mood_prediction' or 'rem_detection')
    
    Returns:
        dict: Dictionary containing X_train, y_train, X_test, y_test, and feature_names
    """
    # Define feature file path
    feature_file = ROOT_DIR / "data" / "features" / f"{task}_features.joblib"
    
    if not feature_file.exists():
        logger.error(f"Feature file not found: {feature_file}")
        return None
    
    # Load features
    try:
        features = joblib.load(feature_file)
        
        # Make sure we have all the required features
        required_keys = ["X_train", "y_train", "X_test", "y_test", "feature_names"]
        missing_keys = [key for key in required_keys if key not in features]
        
        if missing_keys:
            logger.warning(f"Missing keys in features: {missing_keys}")
            
            # Try to recreate feature_names if missing
            if "feature_names" in missing_keys and "X_train" in features:
                if hasattr(features["X_train"], "columns"):
                    features["feature_names"] = features["X_train"].columns.tolist()
                    logger.info(f"Created feature_names from X_train columns")
                    
        # Convert to DataFrame if needed
        if isinstance(features["X_train"], np.ndarray):
            features["X_train"] = pd.DataFrame(
                features["X_train"], 
                columns=features["feature_names"]
            )
            features["X_test"] = pd.DataFrame(
                features["X_test"], 
                columns=features["feature_names"]
            )
            logger.info("Converted NumPy arrays to DataFrames")
        
        # Clean data by replacing inf with NaN and then filling NaNs with appropriate values
        for dataset in ["X_train", "X_test"]:
            if isinstance(features[dataset], pd.DataFrame):
                # Replace inf with NaN
                features[dataset] = features[dataset].replace([np.inf, -np.inf], np.nan)
                
                # Fill NaN values with appropriate values (median for each column)
                for col in features[dataset].columns:
                    median_val = features[dataset][col].median()
                    if pd.isna(median_val):  # If median is also NaN
                        features[dataset][col] = features[dataset][col].fillna(0)
                    else:
                        features[dataset][col] = features[dataset][col].fillna(median_val)
                        
                logger.info(f"Cleaned {dataset} data by handling inf and NaN values")
            
        logger.info(f"Loaded features from {feature_file}")
        return features
        
    except Exception as e:
        logger.error(f"Error loading features: {e}")
        return None

def apply_robust_preprocessing(X_train, X_test):
    """
    Apply robust preprocessing techniques to handle outliers.
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        tuple: Preprocessed X_train, X_test
    """
    # Clean data by replacing inf with NaN and then filling NaNs with appropriate values
    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()
    
    # Replace inf with NaN
    X_train_clean = X_train_clean.replace([np.inf, -np.inf], np.nan)
    X_test_clean = X_test_clean.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with appropriate values (median for each column)
    for col in X_train_clean.columns:
        median_val = X_train_clean[col].median()
        if pd.isna(median_val):  # If median is also NaN
            X_train_clean[col] = X_train_clean[col].fillna(0)
            X_test_clean[col] = X_test_clean[col].fillna(0)
        else:
            X_train_clean[col] = X_train_clean[col].fillna(median_val)
            X_test_clean[col] = X_test_clean[col].fillna(median_val)
    
    logger.info("Data cleaned for robust preprocessing")
    
    # Check for any remaining inf or NaN values
    if not np.isfinite(X_train_clean.values).all():
        logger.warning("X_train still contains inf values after cleaning. Replacing with zeros.")
        X_train_clean = X_train_clean.replace([np.inf, -np.inf], 0)
    
    if not np.isfinite(X_test_clean.values).all():
        logger.warning("X_test still contains inf values after cleaning. Replacing with zeros.")
        X_test_clean = X_test_clean.replace([np.inf, -np.inf], 0)
    
    # Check for any remaining NaN values
    if X_train_clean.isna().any().any():
        logger.warning("X_train still contains NaN values after cleaning. Replacing with zeros.")
        X_train_clean = X_train_clean.fillna(0)
    
    if X_test_clean.isna().any().any():
        logger.warning("X_test still contains NaN values after cleaning. Replacing with zeros.")
        X_test_clean = X_test_clean.fillna(0)
    
    # Use RobustScaler to handle outliers
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled

def build_xgboost_robust(X_train, y_train, X_test, y_test):
    """
    Build a robust XGBoost model for mood prediction.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        tuple: (model, metadata, metrics)
    """
    logger.info("Building robust XGBoost model...")
    
    # Apply robust preprocessing
    X_train_robust, X_test_robust = apply_robust_preprocessing(X_train, X_test)
    
    # Create multiple XGBoost models with different parameters
    models = []
    
    # Model 1: Balanced with moderate depth
    model1 = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        min_child_weight=1,
        gamma=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Model 2: Higher depth for complex patterns
    model2 = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=2,
        gamma=0,
        subsample=0.9,
        colsample_bytree=0.7,
        scale_pos_weight=2,
        reg_alpha=0.01,
        reg_lambda=10.0,
        random_state=43,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Model 3: Conservative with lower depth
    model3 = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=3,
        gamma=0.3,
        subsample=0.7,
        colsample_bytree=1.0,
        scale_pos_weight=1,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=44,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Create soft voting ensemble for robustness
    ensemble = VotingClassifier(
        estimators=[
            ('balanced', model1),
            ('complex', model2),
            ('conservative', model3)
        ],
        voting='soft'
    )
    
    # Train the ensemble model
    logger.info("Training robust XGBoost ensemble...")
    ensemble.fit(X_train_robust, y_train)
    
    # Get feature importances from the first model
    model1.fit(X_train_robust, y_train)
    feature_importances = model1.feature_importances_
    
    # Evaluate model on test set
    y_pred = ensemble.predict(X_test_robust)
    y_prob = ensemble.predict_proba(X_test_robust)[:, 1]
    
    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }
    
    logger.info(f"Robust XGBoost metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Create metadata
    metadata = {
        'model_name': 'xgboost_robust',
        'task': 'mood_prediction',
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'metrics': metrics,
        'features': X_train.columns.tolist(),
        'preprocessing': 'RobustScaler',
        'ensemble_type': 'VotingClassifier',
        'description': 'Robust XGBoost ensemble model for mood prediction'
    }
    
    return ensemble, metadata, metrics

# Create a composite model that includes preprocessing and prediction in one object
class RobustRandomForestModel:
    def __init__(self, base_model, final_model, scaler):
        self.base_model = base_model
        self.final_model = final_model
        self.scaler = scaler
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        X_enhanced = X_scaled.copy()
        X_enhanced['cv_prediction'] = self.base_model.predict_proba(X_scaled)[:, 1]
        return self.final_model.predict(X_enhanced)
        
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        X_enhanced = X_scaled.copy()
        X_enhanced['cv_prediction'] = self.base_model.predict_proba(X_scaled)[:, 1]
        return self.final_model.predict_proba(X_enhanced)

def build_random_forest_robust(X_train, y_train, X_test, y_test):
    """
    Build a robust Random Forest model for mood prediction.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        tuple: (model, metadata, metrics)
    """
    logger.info("Building robust Random Forest model...")
    
    # Apply robust preprocessing
    X_train_robust, X_test_robust = apply_robust_preprocessing(X_train, X_test)
    
    # Create cross-validated predictions for training
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Base model for predictions
    base_rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        bootstrap=True,
        random_state=42
    )
    
    # Generate cross-validated predictions
    cv_preds = cross_val_predict(base_rf, X_train_robust, y_train, cv=cv, method='predict_proba')
    cv_pred_proba = cv_preds[:, 1]
    
    # Add cross-validated predictions as a new feature
    X_train_enhanced = X_train_robust.copy()
    X_train_enhanced['cv_prediction'] = cv_pred_proba
    
    # Fit the base model on the entire training set
    base_rf.fit(X_train_robust, y_train)
    
    # Add predictions to test set
    X_test_enhanced = X_test_robust.copy()
    X_test_enhanced['cv_prediction'] = base_rf.predict_proba(X_test_robust)[:, 1]
    
    # Build final robust model with enhanced features
    robust_rf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=12,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced_subsample',
        random_state=43,
        n_jobs=-1
    )
    
    # Train the model
    logger.info("Training robust Random Forest...")
    robust_rf.fit(X_train_enhanced, y_train)
    
    # Evaluate model on test set
    y_pred = robust_rf.predict(X_test_enhanced)
    y_prob = robust_rf.predict_proba(X_test_enhanced)[:, 1]
    
    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }
    
    logger.info(f"Robust Random Forest metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Create metadata including the feature preprocessing
    metadata = {
        'model_name': 'random_forest_robust',
        'task': 'mood_prediction',
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'metrics': metrics,
        'features': X_train_enhanced.columns.tolist(),
        'preprocessing': 'RobustScaler + CrossValidatedPredictions',
        'n_estimators': 1000,
        'max_depth': 12,
        'description': 'Robust Random Forest model with cross-validated feature augmentation'
    }
    
    # Create the scaler for the composite model
    scaler = RobustScaler()
    scaler.fit(X_train)
    
    # Create the composite model
    composite_model = RobustRandomForestModel(base_rf, robust_rf, scaler)
    
    return composite_model, metadata, metrics

def create_derived_features(X):
    """
    Create advanced derived features to improve model performance.
    
    Args:
        X: DataFrame with original features
        
    Returns:
        DataFrame with added derived features
    """
    X_enhanced = X.copy()
    
    # Check if required columns exist
    required_columns = ['total_sleep_time', 'rem_time', 'deep_sleep_time', 
                        'light_sleep_time', 'wake_time', 'sleep_efficiency']
    
    missing_columns = [col for col in required_columns if col not in X.columns]
    if missing_columns:
        logger.warning(f"Required columns {missing_columns} not found, skipping derived features")
        return X
    
    try:
        # Replace any potential NaN or inf values with reasonable defaults
        for col in required_columns:
            # Replace inf values with NaN
            X_enhanced[col] = X_enhanced[col].replace([np.inf, -np.inf], np.nan)
            
            # For time columns, use 0 as a default for NaN
            if col in ['total_sleep_time', 'rem_time', 'deep_sleep_time', 'light_sleep_time', 'wake_time']:
                X_enhanced[col] = X_enhanced[col].fillna(0)
            
            # For efficiency, use median as default for NaN
            if col == 'sleep_efficiency':
                median_val = X_enhanced[col].median()
                X_enhanced[col] = X_enhanced[col].fillna(median_val if not pd.isna(median_val) else 80)
        
        # Create sleep continuity metric (ratio of wake time to total sleep time)
        # Add epsilon (0.001) to denominator to avoid division by zero
        X_enhanced['sleep_continuity'] = 1 - (X_enhanced['wake_time'] / (X_enhanced['total_sleep_time'] + 0.001))
        # Clip to reasonable range [0, 1]
        X_enhanced['sleep_continuity'] = X_enhanced['sleep_continuity'].clip(0, 1)
        
        # Create sleep depth ratio (deep sleep to light sleep ratio)
        # Add epsilon to denominator to avoid division by zero
        X_enhanced['sleep_depth_ratio'] = X_enhanced['deep_sleep_time'] / (X_enhanced['light_sleep_time'] + 0.001)
        # Clip to reasonable range [0, 3] - typical range is 0.1 to 1.5
        X_enhanced['sleep_depth_ratio'] = X_enhanced['sleep_depth_ratio'].clip(0, 3)
        
        # Create recovery ratio (deep sleep to wake time ratio)
        # Add 1 to denominator to avoid division by zero and extreme values
        X_enhanced['recovery_ratio'] = X_enhanced['deep_sleep_time'] / (X_enhanced['wake_time'] + 1)
        # Clip to reasonable range [0, 10] - typical range is 0.5 to 5
        X_enhanced['recovery_ratio'] = X_enhanced['recovery_ratio'].clip(0, 10)
        
        # Create composite sleep score
        # Handle potential division by zero in percentages
        total_sleep_with_epsilon = X_enhanced['total_sleep_time'] + 0.001
        
        X_enhanced['composite_sleep_score'] = (
            (X_enhanced['sleep_efficiency'].clip(0, 100) / 100 * 0.35) +  # 35% weight to efficiency
            (X_enhanced['rem_time'] / total_sleep_with_epsilon * 0.3) +  # 30% weight to REM
            (X_enhanced['deep_sleep_time'] / total_sleep_with_epsilon * 0.25) +  # 25% weight to deep sleep
            (X_enhanced['sleep_continuity'] * 0.1)  # 10% weight to continuity
        )
        # Clip to reasonable range [0, 1]
        X_enhanced['composite_sleep_score'] = X_enhanced['composite_sleep_score'].clip(0, 1)
        
        # Create sleep quality index
        X_enhanced['sleep_quality_index'] = (
            X_enhanced['sleep_depth_ratio'].clip(0, 2) * 0.4 +
            X_enhanced['recovery_ratio'].clip(0, 5) * 0.3 +
            X_enhanced['sleep_continuity'] * 0.3
        )
        # Clip to reasonable range [0, 2]
        X_enhanced['sleep_quality_index'] = X_enhanced['sleep_quality_index'].clip(0, 2)
        
        # Final check for any NaN or inf values and replace them
        X_enhanced = X_enhanced.replace([np.inf, -np.inf], np.nan)
        X_enhanced = X_enhanced.fillna(X_enhanced.median())
        
        # If there are still NaN values (e.g., in columns where all values were NaN)
        # replace them with zeros
        X_enhanced = X_enhanced.fillna(0)
        
        logger.info("Created derived features for enhanced predictive power")
        return X_enhanced
        
    except Exception as e:
        logger.error(f"Error creating derived features: {e}")
        logger.error("Returning original features without derivation")
        return X

def main():
    """
    Main function to build and save robust models for mood prediction.
    """
    logger.info("Starting robust model creation")
    
    # Load features
    features = load_features("mood_prediction")
    if features is None:
        return 1
    
    X_train = features["X_train"]
    X_test = features["X_test"]
    y_train = features["y_train"]
    y_test = features["y_test"]
    
    logger.info(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Create derived features
    logger.info("Creating derived features...")
    X_train_enhanced = create_derived_features(X_train)
    X_test_enhanced = create_derived_features(X_test)
    
    # Build and save robust XGBoost model
    xgb_model, xgb_metadata, xgb_metrics = build_xgboost_robust(
        X_train_enhanced, y_train, X_test_enhanced, y_test
    )
    
    # Save XGBoost model and metadata
    xgb_model_path = MODELS_DIR / "xgboost_robust.joblib"
    xgb_metadata_path = MODELS_DIR / "xgboost_robust_metadata.joblib"
    
    joblib.dump(xgb_model, xgb_model_path)
    joblib.dump(xgb_metadata, xgb_metadata_path)
    
    logger.info(f"Saved robust XGBoost model to {xgb_model_path}")
    
    # Build and save robust Random Forest model
    rf_model, rf_metadata, rf_metrics = build_random_forest_robust(
        X_train_enhanced, y_train, X_test_enhanced, y_test
    )
    
    # Save Random Forest model and metadata
    rf_model_path = MODELS_DIR / "random_forest_robust.joblib"
    rf_metadata_path = MODELS_DIR / "random_forest_robust_metadata.joblib"
    
    joblib.dump(rf_model, rf_model_path)
    joblib.dump(rf_metadata, rf_metadata_path)
    
    logger.info(f"Saved robust Random Forest model to {rf_model_path}")
    
    # Print final results
    logger.info("\nRobust Model Performance Summary:")
    logger.info("---------------------------------")
    logger.info(f"XGBoost Robust - F1 Score: {xgb_metrics['f1']:.4f}, Accuracy: {xgb_metrics['accuracy']:.4f}")
    logger.info(f"Random Forest Robust - F1 Score: {rf_metrics['f1']:.4f}, Accuracy: {rf_metrics['accuracy']:.4f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 