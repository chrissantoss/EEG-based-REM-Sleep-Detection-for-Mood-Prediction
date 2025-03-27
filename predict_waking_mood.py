#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script predicts mood based on sleep quality metrics using the trained model.
It provides sample sleep quality metrics and predicts the corresponding waking mood.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

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
        
        # Get selected features
        selected_features = metadata.get("selected_features", [])
        
        if selected_features:
            # Check if all required features are present
            missing_features = [f for f in selected_features if f not in metrics_df.columns]
            
            if missing_features:
                logger.error(f"Missing features for mood prediction: {missing_features}")
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

def main():
    """Main function to predict mood from sleep metrics."""
    logger.info("Predicting mood based on different sleep metrics scenarios")
    
    # Generate sample sleep metrics
    sample_metrics = generate_sample_sleep_metrics()
    
    # Predict mood for each sample
    models = ["xgboost", "random_forest", "svm", "logistic_regression"]
    
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

if __name__ == "__main__":
    main() 