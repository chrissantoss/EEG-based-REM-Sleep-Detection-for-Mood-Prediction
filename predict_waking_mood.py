#!/usr/bin/env python
# -*- coding: utf-8 -*-


# This script predicts mood based on sleep quality metrics using the trained model.
# It provides sample sleep quality metrics and predicts the corresponding waking mood.


import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Define the data directory
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"

def load_model(model_name="xgboost", task="mood_prediction"):
    """
    Load a trained model.
    
    Args:
        model_name (str): Name of the model
        task (str): Task name ('rem_detection' or 'mood_prediction')
    
    Returns:
        object: Trained model
        dict: Model metadata
    """
    # Determine model file paths
    model_dir = MODELS_DIR / task
    model_file = model_dir / f"{model_name}.joblib"
    metadata_file = model_dir / f"{model_name}_metadata.joblib"
    
    # Check if files exist
    if not model_file.exists() or not metadata_file.exists():
        logger.error(f"Model or metadata file not found: {model_file}, {metadata_file}")
        return None, None
    
    try:
        # Load model and metadata
        model = joblib.load(model_file)
        metadata = joblib.load(metadata_file)
        
        logger.info(f"Loaded {model_name} model for {task}")
        return model, metadata
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

def generate_sample_sleep_metrics():
    """
    Generate a range of sample sleep metrics for demonstration.
    
    Returns:
        list: List of sample sleep metrics dictionaries
    """
    # We'll generate a range of sleep metrics based on known relationships
    samples = []
    
    # Full metrics list required by the model:
    # 'total_sleep_time', 'wake_time', 'rem_time', 'light_sleep_time', 
    # 'deep_sleep_time', 'sleep_efficiency', 'rem_percentage', 'rem_cycles', 'rem_awakenings'
    
    # Scenario 1: Optimal sleep (high REM percentage, few awakenings)
    samples.append({
        'total_sleep_time': 480,  # 8 hours in minutes
        'wake_time': 20,          # 20 minutes awake during the night
        'rem_time': 120,          # 2 hours of REM
        'light_sleep_time': 240,  # 4 hours of light sleep
        'deep_sleep_time': 120,   # 2 hours of deep sleep
        'sleep_efficiency': 96,   # 96% sleep efficiency (480/500)
        'rem_percentage': 25,     # 25% REM (optimal)
        'rem_cycles': 4,          # 4 complete cycles
        'rem_awakenings': 1       # Few awakenings
    })
    
    # Scenario 2: Poor sleep (low REM percentage, many awakenings)
    samples.append({
        'total_sleep_time': 360,  # 6 hours
        'wake_time': 60,          # 1 hour awake during the night
        'rem_time': 54,           # 54 min of REM (15%)
        'light_sleep_time': 216,  # 3.6 hours of light sleep
        'deep_sleep_time': 90,    # 1.5 hours of deep sleep
        'sleep_efficiency': 85,   # 85% sleep efficiency
        'rem_percentage': 15,     # Low REM percentage
        'rem_cycles': 3,          # 3 cycles
        'rem_awakenings': 5       # Many awakenings
    })
    
    # Scenario 3: Very poor sleep (very low REM percentage, many awakenings)
    samples.append({
        'total_sleep_time': 240,  # 4 hours
        'wake_time': 90,          # 1.5 hours awake
        'rem_time': 24,           # 24 min of REM
        'light_sleep_time': 156,  # 2.6 hours of light sleep
        'deep_sleep_time': 60,    # 1 hour of deep sleep
        'sleep_efficiency': 72,   # 72% sleep efficiency (240/330)
        'rem_percentage': 10,     # Very low REM percentage
        'rem_cycles': 2,          # 2 incomplete cycles
        'rem_awakenings': 7       # Many awakenings
    })
    
    # Scenario 4: Long but disturbed sleep
    samples.append({
        'total_sleep_time': 540,  # 9 hours
        'wake_time': 60,          # 1 hour awake
        'rem_time': 81,           # 81 min of REM
        'light_sleep_time': 324,  # 5.4 hours of light sleep
        'deep_sleep_time': 135,   # 2.25 hours of deep sleep
        'sleep_efficiency': 90,   # 90% sleep efficiency (540/600)
        'rem_percentage': 15,     # Low REM percentage
        'rem_cycles': 5,          # 5 cycles
        'rem_awakenings': 6       # Many awakenings
    })
    
    # Scenario 5: Short but high quality sleep
    samples.append({
        'total_sleep_time': 360,  # 6 hours
        'wake_time': 10,          # Only 10 min awake
        'rem_time': 90,           # 90 min of REM
        'light_sleep_time': 180,  # 3 hours of light sleep
        'deep_sleep_time': 90,    # 1.5 hours of deep sleep
        'sleep_efficiency': 97,   # 97% sleep efficiency (360/370)
        'rem_percentage': 25,     # High REM percentage
        'rem_cycles': 4,          # 4 cycles
        'rem_awakenings': 1       # Few awakenings
    })
    
    return samples

def create_derived_features(metrics_df):
    """
    Create advanced derived features needed by robust models.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame containing sleep metrics
        
    Returns:
        pd.DataFrame: DataFrame with added derived features
    """
    logger.info("Creating derived features for robust models")
    X_enhanced = metrics_df.copy()
    
    # Check if required columns exist
    required_columns = ['total_sleep_time', 'rem_time', 'deep_sleep_time', 
                        'light_sleep_time', 'wake_time', 'sleep_efficiency']
    
    missing_columns = [col for col in required_columns if col not in metrics_df.columns]
    if missing_columns:
        logger.warning(f"Required columns {missing_columns} not found, skipping derived features")
        return metrics_df
    
    try:
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
            (X_enhanced['sleep_efficiency'] / 100 * 0.35) +  # 35% weight to efficiency
            (X_enhanced['rem_time'] / total_sleep_with_epsilon * 0.3) +  # 30% weight to REM
            (X_enhanced['deep_sleep_time'] / total_sleep_with_epsilon * 0.25) +  # 25% weight to deep sleep
            (X_enhanced['sleep_continuity'] * 0.1)  # 10% weight to continuity
        )
        # Clip to reasonable range [0, 1]
        X_enhanced['composite_sleep_score'] = X_enhanced['composite_sleep_score'].clip(0, 1)
        
        # Create sleep quality index
        X_enhanced['sleep_quality_index'] = (
            X_enhanced['sleep_depth_ratio'] * 0.4 +
            X_enhanced['recovery_ratio'] * 0.3 +
            X_enhanced['sleep_continuity'] * 0.3
        )
        # Clip to reasonable range [0, 2]
        X_enhanced['sleep_quality_index'] = X_enhanced['sleep_quality_index'].clip(0, 2)
        
        # Add default values for the advanced features expected by robust models
        if 'stress_level' not in X_enhanced.columns:
            X_enhanced['stress_level'] = 5  # Default moderate stress level (range 0-10)
            
        if 'exercise_minutes' not in X_enhanced.columns:
            X_enhanced['exercise_minutes'] = 30  # Default moderate exercise (30 minutes)
            
        if 'caffeine_mg' not in X_enhanced.columns:
            X_enhanced['caffeine_mg'] = 100  # Default moderate caffeine (equivalent to ~1 cup of coffee)
            
        if 'alcohol_units' not in X_enhanced.columns:
            X_enhanced['alcohol_units'] = 0  # Default no alcohol
            
        # Calculate stress-exercise balance
        if 'stress_level' in X_enhanced.columns and 'exercise_minutes' in X_enhanced.columns:
            X_enhanced['stress_exercise_balance'] = X_enhanced['exercise_minutes'] / (X_enhanced['stress_level'] * 10 + 0.001)
            X_enhanced['stress_exercise_balance'] = X_enhanced['stress_exercise_balance'].clip(0, 5)
            
        logger.info("Created derived features for robust model prediction")
        logger.info(f"Added features: {set(X_enhanced.columns) - set(metrics_df.columns)}")
        
        return X_enhanced
        
    except Exception as e:
        logger.error(f"Error creating derived features: {e}")
        logger.error("Returning original features without derivation")
        return metrics_df

def predict_mood(sleep_metrics, model_name="xgboost"):
    """
    Predict mood based on sleep metrics.
    
    Args:
        sleep_metrics (dict): Sleep metrics
        model_name (str): Name of the model to use
    
    Returns:
        dict: Mood prediction results
    """
    # Load mood prediction model
    model, metadata = load_model(model_name, "mood_prediction")
    
    if model is None or metadata is None:
        return None
    
    try:
        # Convert metrics to DataFrame
        metrics_df = pd.DataFrame([sleep_metrics])
        
        # Add derived features for robust models
        if "robust" in model_name:
            metrics_df = create_derived_features(metrics_df)
        
        # Get selected features from metadata
        if "features" in metadata:
            selected_features = metadata["features"]
        else:
            selected_features = metadata.get("selected_features", [])
        
        if selected_features:
            # Check if all required features are present
            missing_features = [f for f in selected_features if f not in metrics_df.columns]
            
            if missing_features:
                logger.error(f"Missing features for {model_name}: {missing_features}")
                return None
            
            # Select features used by the model
            X = metrics_df[selected_features]
        else:
            # If no selected features in metadata, use all features
            X = metrics_df
        
        # Make predictions
        mood_pred = model.predict(X)[0]
        mood_prob = model.predict_proba(X)[0, 1]
        
        # Create result dictionary
        result = {
            'good_mood': bool(mood_pred),
            'good_mood_probability': mood_prob,
            'mood_score': mood_prob * 10.0  # Scale to 0-10
        }
        
        logger.info(f"Mood prediction: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error predicting mood: {e}")
        return None

def main():
    """Main function to predict mood from sleep metrics."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Predict mood based on sleep metrics")
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=[
            "xgboost", "random_forest", "svm", "logistic_regression", 
            "xgboost_robust", "random_forest_robust"
        ],
        help="Model to use for prediction (default: xgboost)"
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=0,
        help="Specific scenario to test (0 for all scenarios, 1-5 for individual scenarios)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    args = parser.parse_args()
    
    logger.info(f"Predicting mood using the {args.model} model")
    
    # Generate sample sleep metrics
    sample_metrics = generate_sample_sleep_metrics()
    
    # Filter scenarios if specified
    if args.scenario > 0 and args.scenario <= len(sample_metrics):
        sample_metrics = [sample_metrics[args.scenario - 1]]
        logger.info(f"Testing only scenario {args.scenario}")
    
    # Predict mood for each sample
    if args.scenario == 0:
        # If testing all scenarios, use the specified model
        models = [args.model]
    else:
        # If testing a specific scenario, use all models for comparison
        if args.verbose:
            models = [
                "xgboost", "random_forest", "svm", "logistic_regression", 
                "xgboost_robust", "random_forest_robust"
            ]
        else:
            models = [args.model]
    
    for i, metrics in enumerate(sample_metrics):
        logger.info(f"\nScenario {i+1}:")
        logger.info(f"Sleep Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        
        logger.info(f"\nMood Predictions:")
        for model_name in models:
            logger.info(f"\nUsing {model_name} model:")
            result = predict_mood(metrics, model_name)
            
            if result:
                logger.info(f"  Predicted Mood: {'Good' if result['good_mood'] else 'Bad'}")
                logger.info(f"  Confidence: {result['good_mood_probability']:.2f}")
                logger.info(f"  Mood Score (0-10): {result['mood_score']:.1f}")
            else:
                logger.info(f"  Failed to predict mood using {model_name}")
                
                # If this is a robust model that failed, try looking in the robust_models directory
                if "robust" in model_name:
                    robust_model_path = MODELS_DIR / "robust_models" / f"{model_name}.joblib"
                    robust_metadata_path = MODELS_DIR / "robust_models" / f"{model_name}_metadata.joblib"
                    
                    if robust_model_path.exists() and robust_metadata_path.exists():
                        logger.info(f"Trying to load robust model from alternate location: {robust_model_path}")
                        try:
                            model = joblib.load(robust_model_path)
                            metadata = joblib.load(robust_metadata_path)
                            
                            # Convert metrics to DataFrame
                            metrics_df = pd.DataFrame([metrics])
                            
                            # Get selected features
                            selected_features = metadata.get("selected_features", [])
                            
                            if selected_features:
                                # Check if all required features are present
                                missing_features = [f for f in selected_features if f not in metrics_df.columns]
                                
                                if missing_features:
                                    logger.error(f"Missing features for mood prediction: {missing_features}")
                                    continue
                                
                                # Select features used by the model
                                X = metrics_df[selected_features]
                            else:
                                # If no selected features in metadata, use all features
                                X = metrics_df
                            
                            # Make predictions
                            mood_pred = model.predict(X)[0]
                            mood_prob = model.predict_proba(X)[0, 1]
                            
                            # Create result dictionary
                            result = {
                                'good_mood': bool(mood_pred),
                                'good_mood_probability': mood_prob,
                                'mood_score': mood_prob * 10.0  # Scale to 0-10
                            }
                            
                            logger.info(f"  Predicted Mood: {'Good' if result['good_mood'] else 'Bad'}")
                            logger.info(f"  Confidence: {result['good_mood_probability']:.2f}")
                            logger.info(f"  Mood Score (0-10): {result['mood_score']:.1f}")
                        except Exception as e:
                            logger.error(f"Error with robust model from alternate location: {e}")
                
    logger.info("\nCompleted mood predictions")

if __name__ == "__main__":
    main() 