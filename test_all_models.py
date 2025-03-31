#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Tests all trained mood prediction models using consistent metrics.
# Compares accuracy, precision, recall, F1, and ROC AUC across models.

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns

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
ENHANCED_DIR = MODELS_DIR / "enhanced_models"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_models():
    # Loads all available models from both base and enhanced directories
    models = {}
    
    # Load base mood prediction models
    base_model_dir = MODELS_DIR / "mood_prediction"
    if base_model_dir.exists():
        # Get all model files
        model_files = list(base_model_dir.glob("*.joblib"))
        
        for model_file in model_files:
            if "_metadata" in model_file.name:
                continue
                
            model_name = model_file.stem
            metadata_file = base_model_dir / f"{model_name}_metadata.joblib"
            
            if not metadata_file.exists():
                logger.warning(f"Metadata file not found for {model_name}")
                continue
                
            try:
                model = joblib.load(model_file)
                metadata = joblib.load(metadata_file)
                
                models[model_name] = {
                    "model": model,
                    "metadata": metadata,
                    "file_path": str(model_file),
                    "type": "base"
                }
                
                logger.info(f"Loaded base model: {model_name}")
                
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
    
    # Load enhanced models
    if ENHANCED_DIR.exists():
        # Get all enhanced model files
        enhanced_model_files = list(ENHANCED_DIR.glob("*.joblib"))
        
        for model_file in enhanced_model_files:
            if "_metadata" in model_file.name:
                continue
                
            model_name = model_file.stem
            metadata_file = ENHANCED_DIR / f"{model_name}_metadata.joblib"
            
            if not metadata_file.exists():
                logger.warning(f"Metadata file not found for enhanced model {model_name}")
                continue
                
            try:
                model = joblib.load(model_file)
                metadata = joblib.load(metadata_file)
                
                models[model_name] = {
                    "model": model,
                    "metadata": metadata,
                    "file_path": str(model_file),
                    "type": "enhanced"
                }
                
                logger.info(f"Loaded enhanced model: {model_name}")
                
            except Exception as e:
                logger.error(f"Error loading enhanced model {model_name}: {e}")
    
    # Load robust models (look in both base and robust directories)
    # First check in base_model_dir for models with "robust" in the name
    robust_model_files = [f for f in base_model_dir.glob("*robust*.joblib") if "_metadata" not in f.name]
    
    # Also check in the robust_models directory if it exists
    robust_dir = ROOT_DIR / "models" / "robust_models"
    if robust_dir.exists():
        robust_model_files.extend([f for f in robust_dir.glob("*.joblib") if "_metadata" not in f.name])
    
    for model_file in robust_model_files:
        if "_metadata" in model_file.name:
            continue
            
        model_name = model_file.stem
        metadata_file = model_file.parent / f"{model_name}_metadata.joblib"
        
        if not metadata_file.exists():
            logger.warning(f"Metadata file not found for robust model {model_name}")
            continue
            
        try:
            model = joblib.load(model_file)
            metadata = joblib.load(metadata_file)
            
            models[model_name] = {
                "model": model,
                "metadata": metadata,
                "file_path": str(model_file),
                "type": "robust"
            }
            
            logger.info(f"Loaded robust model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading robust model {model_name}: {e}")
    
    logger.info(f"Loaded {len(models)} models in total")
    return models

def load_test_data():
    # Loads test data for model evaluation
    test_data_file = FEATURES_DIR / "mood_prediction_features.joblib"
    
    if not test_data_file.exists():
        logger.error(f"Test data file not found: {test_data_file}")
        return None
    
    try:
        features = joblib.load(test_data_file)
        X_test = features["X_test"]
        y_test = features["y_test"]
        feature_names = features["feature_names"]
        
        # Convert NumPy array to pandas DataFrame with feature names
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test, columns=feature_names)
            logger.info("Converted NumPy array to pandas DataFrame")
        
        # Log available features for debugging
        logger.info(f"Loaded test data with {len(X_test)} samples and {len(feature_names)} features")
        logger.info(f"Available features in test data: {list(X_test.columns)}")
        
        return {
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": feature_names
        }
    
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return None

def evaluate_model(model, X_test, y_test, model_name):
    # Evaluates model performance on test data and returns metrics
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob))
        }
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Evaluation results for {model_name}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return {
            "metrics": metrics,
            "predictions": y_pred.tolist(),
            "probabilities": y_prob.tolist(),
            "confusion_matrix": conf_matrix.tolist()
        }
    
    except Exception as e:
        logger.error(f"Error evaluating model {model_name}: {e}")
        return None

def test_all_models():
    # Tests all models with the same test dataset and gathers performance metrics
    # Load models
    models = load_models()
    if not models:
        logger.error("No models found")
        return None
    
    # Load test data
    test_data = load_test_data()
    if test_data is None:
        logger.error("Failed to load test data")
        return None
    
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    feature_names = test_data["feature_names"]
    
    # Test each model
    results = {}
    for model_name, model_info in models.items():
        logger.info(f"\nTesting model: {model_name}")
        
        model = model_info["model"]
        model_type = model_info["type"]
        
        # Check if model features match test data
        expected_features = None
        
        # Get expected features from metadata
        if model_type == "base" and "features" in model_info["metadata"]:
            expected_features = model_info["metadata"]["features"]
        elif model_type == "base" and "selected_features" in model_info["metadata"]:
            expected_features = model_info["metadata"]["selected_features"]
        elif model_type == "enhanced" and "features" in model_info["metadata"]:
            expected_features = model_info["metadata"]["features"]
        
        if expected_features:
            logger.info(f"Model {model_name} expects these features: {expected_features}")
            
            # Check for feature discrepancies
            available_features = set(X_test.columns)
            required_features = set(expected_features)
            
            # Find missing and extra features
            missing_features = required_features - available_features
            common_features = required_features.intersection(available_features)
            
            if missing_features:
                logger.warning(f"Missing features for {model_name}: {missing_features}")
                logger.warning("Cannot evaluate this model correctly - skipping")
                continue
            
            # If we have all needed features, select only those for evaluation
            if len(common_features) == len(required_features):
                logger.info(f"Using {len(expected_features)} features for model evaluation")
                model_X_test = X_test[expected_features]
                
                # Evaluate model
                evaluation = evaluate_model(model, model_X_test, y_test, model_name)
                
                if evaluation:
                    results[model_name] = {
                        "evaluation": evaluation,
                        "model_type": model_type,
                        "file_path": model_info["file_path"]
                    }
            else:
                logger.warning(f"Cannot use model {model_name} - missing required features")
        else:
            logger.warning(f"No feature information found for {model_name}, cannot evaluate safely")
    
    if not results:
        logger.error("No models were successfully evaluated")
        return None
    
    # Save results
    results_file = RESULTS_DIR / "model_comparison.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved model comparison results to {results_file}")
    
    # Create visualization of results
    visualize_results(results)
    
    return results

def visualize_results(results):
    # Creates visualizations of model performance metrics for easy comparison
    if not results:
        logger.error("No results to visualize")
        return
    
    # Extract metrics for all models
    model_names = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    model_types = [results[name]["model_type"] for name in model_names]
    
    # Create DataFrame for plotting
    data = []
    for model_name in model_names:
        model_metrics = results[model_name]["evaluation"]["metrics"]
        model_type = results[model_name]["model_type"]
        
        for metric in metrics:
            data.append({
                "Model": model_name,
                "Metric": metric,
                "Value": model_metrics[metric],
                "Type": model_type
            })
    
    df = pd.DataFrame(data)
    
    # Plot metrics by model
    plt.figure(figsize=(12, 8))
    
    # Use different colors for base vs enhanced models
    palette = {"base": "steelblue", "enhanced": "darkorange"}
    
    sns.barplot(x="Model", y="Value", hue="Metric", data=df)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save the plot
    metrics_plot_path = RESULTS_DIR / "model_metrics_comparison.png"
    plt.savefig(metrics_plot_path)
    logger.info(f"Saved metrics comparison plot to {metrics_plot_path}")
    
    # Plot confusion matrices
    n_models = len(model_names)
    fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, model_name in enumerate(model_names):
        conf_matrix = np.array(results[model_name]["evaluation"]["confusion_matrix"])
        
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=axes[i])
        axes[i].set_title(f"{model_name}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
    
    # Hide any unused subplots
    for i in range(n_models, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    # Save the plot
    confusion_plot_path = RESULTS_DIR / "model_confusion_matrices.png"
    plt.savefig(confusion_plot_path)
    logger.info(f"Saved confusion matrices plot to {confusion_plot_path}")

def main():
    # Main function to test and compare all available mood prediction models
    logger.info("Starting model testing and comparison")
    
    # Test all models
    results = test_all_models()
    
    if results:
        # Find best model
        best_model = None
        best_f1 = 0
        
        for model_name, model_data in results.items():
            f1 = model_data["evaluation"]["metrics"]["f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_model = model_name
        
        if best_model:
            logger.info(f"\nBest model by F1 score: {best_model} (F1: {best_f1:.4f})")
            
            # Create symlink to best model
            best_model_path = results[best_model]["file_path"]
            best_model_link = MODELS_DIR / "best_mood_model.joblib"
            
            try:
                if best_model_link.exists():
                    best_model_link.unlink()
                
                # Use relative path for symlink
                best_model_path = Path(best_model_path)
                best_model_link.symlink_to(best_model_path.relative_to(best_model_link.parent.parent))
                
                logger.info(f"Created symlink to best model at {best_model_link}")
            except Exception as e:
                logger.error(f"Error creating symlink to best model: {e}")
    
    logger.info("Testing completed")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 