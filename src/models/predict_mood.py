#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script predicts mood based on sleep patterns using trained models.
It can process EEG data or sleep metrics to predict mood upon waking.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import mne
from scipy.signal import welch

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
MODELS_DIR = ROOT_DIR / "models"

def load_model(model_name, task):
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

def process_eeg_file(file_path):
    """
    Process an EEG file to extract features for REM detection.
    
    Args:
        file_path (str): Path to the EEG file
    
    Returns:
        pd.DataFrame: Extracted features
    """
    try:
        # Read the EEG file
        raw = mne.io.read_raw_edf(file_path, preload=True)
        
        # Apply bandpass filter (0.5-30 Hz)
        raw_filtered = raw.copy().filter(l_freq=0.5, h_freq=30.0)
        
        # Get data and sampling frequency
        data, times = raw_filtered.get_data(return_times=True)
        sfreq = raw_filtered.info['sfreq']
        
        # Segment the data (30-second windows)
        window_size = 30.0  # seconds
        window_samples = int(window_size * sfreq)
        
        segments = []
        for start in range(0, data.shape[1] - window_samples + 1, window_samples):
            end = start + window_samples
            segment = data[:, start:end]
            segments.append(segment)
        
        # Extract features
        features = []
        
        for i, segment in enumerate(segments):
            segment_features = {}
            
            # Process each channel
            for ch_idx, channel_data in enumerate(segment):
                # Time domain features
                segment_features[f'mean_ch{ch_idx}'] = np.mean(channel_data)
                segment_features[f'std_ch{ch_idx}'] = np.std(channel_data)
                segment_features[f'min_ch{ch_idx}'] = np.min(channel_data)
                segment_features[f'max_ch{ch_idx}'] = np.max(channel_data)
                segment_features[f'ptp_ch{ch_idx}'] = np.ptp(channel_data)
                
                # Frequency domain features
                freqs, psd = welch(channel_data, fs=sfreq, nperseg=min(256, len(channel_data)))
                
                # Frequency bands
                delta_idx = np.logical_and(freqs >= 0.5, freqs <= 4)
                theta_idx = np.logical_and(freqs >= 4, freqs <= 8)
                alpha_idx = np.logical_and(freqs >= 8, freqs <= 13)
                beta_idx = np.logical_and(freqs >= 13, freqs <= 30)
                
                # Band powers
                segment_features[f'delta_power_ch{ch_idx}'] = np.sum(psd[delta_idx])
                segment_features[f'theta_power_ch{ch_idx}'] = np.sum(psd[theta_idx])
                segment_features[f'alpha_power_ch{ch_idx}'] = np.sum(psd[alpha_idx])
                segment_features[f'beta_power_ch{ch_idx}'] = np.sum(psd[beta_idx])
                
                # Relative band powers
                total_power = np.sum(psd)
                if total_power > 0:
                    segment_features[f'delta_rel_power_ch{ch_idx}'] = np.sum(psd[delta_idx]) / total_power
                    segment_features[f'theta_rel_power_ch{ch_idx}'] = np.sum(psd[theta_idx]) / total_power
                    segment_features[f'alpha_rel_power_ch{ch_idx}'] = np.sum(psd[alpha_idx]) / total_power
                    segment_features[f'beta_rel_power_ch{ch_idx}'] = np.sum(psd[beta_idx]) / total_power
            
            # Add segment index
            segment_features['segment_idx'] = i
            
            features.append(segment_features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features)
        
        logger.info(f"Extracted {features_df.shape[1]} features from {len(segments)} segments")
        return features_df
    
    except Exception as e:
        logger.error(f"Error processing EEG file {file_path}: {e}")
        return None

def detect_rem_sleep(eeg_features, model_name="random_forest"):
    """
    Detect REM sleep stages from EEG features.
    
    Args:
        eeg_features (pd.DataFrame): EEG features
        model_name (str): Name of the model to use
    
    Returns:
        pd.DataFrame: Features with REM sleep predictions
    """
    # Load REM detection model
    model, metadata = load_model(model_name, "rem_detection")
    
    if model is None or metadata is None:
        return None
    
    try:
        # Get selected features
        selected_features = metadata.get("selected_features", [])
        
        # Check if all required features are present
        missing_features = [f for f in selected_features if f not in eeg_features.columns]
        
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            return None
        
        # Select features used by the model
        X = eeg_features[selected_features]
        
        # Make predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        # Add predictions to features
        eeg_features['is_rem'] = y_pred
        eeg_features['rem_probability'] = y_prob
        
        # Count REM segments
        rem_count = np.sum(y_pred)
        total_count = len(y_pred)
        rem_percentage = rem_count / total_count * 100 if total_count > 0 else 0
        
        logger.info(f"Detected {rem_count} REM segments out of {total_count} ({rem_percentage:.1f}%)")
        
        return eeg_features
    
    except Exception as e:
        logger.error(f"Error detecting REM sleep: {e}")
        return None

def extract_sleep_metrics(eeg_features_with_rem):
    """
    Extract sleep metrics from EEG features with REM predictions.
    
    Args:
        eeg_features_with_rem (pd.DataFrame): EEG features with REM predictions
    
    Returns:
        dict: Sleep metrics
    """
    try:
        # Calculate sleep metrics
        total_segments = len(eeg_features_with_rem)
        segment_duration = 30.0  # seconds
        total_sleep_time = total_segments * segment_duration / 60.0  # minutes
        
        # REM sleep metrics
        rem_segments = eeg_features_with_rem[eeg_features_with_rem['is_rem'] == 1]
        rem_count = len(rem_segments)
        rem_time = rem_count * segment_duration / 60.0  # minutes
        rem_percentage = rem_count / total_segments * 100 if total_segments > 0 else 0
        
        # REM fragmentation (number of transitions from REM to non-REM)
        is_rem_array = eeg_features_with_rem['is_rem'].values
        rem_transitions = np.sum(np.diff(is_rem_array) < 0)
        
        # Estimate number of REM cycles
        # A simple heuristic: count sequences of consecutive REM segments
        rem_cycles = 0
        in_rem = False
        for is_rem in is_rem_array:
            if is_rem and not in_rem:
                rem_cycles += 1
                in_rem = True
            elif not is_rem:
                in_rem = False
        
        # Create metrics dictionary
        metrics = {
            'total_sleep_time': total_sleep_time,
            'rem_time': rem_time,
            'rem_percentage': rem_percentage,
            'rem_cycles': rem_cycles,
            'rem_awakenings': rem_transitions
        }
        
        logger.info(f"Extracted sleep metrics: {metrics}")
        return metrics
    
    except Exception as e:
        logger.error(f"Error extracting sleep metrics: {e}")
        return None

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
        
        # Check if all required features are present
        missing_features = [f for f in selected_features if f not in metrics_df.columns]
        
        if missing_features:
            logger.error(f"Missing features for mood prediction: {missing_features}")
            return None
        
        # Select features used by the model
        X = metrics_df[selected_features]
        
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

def process_sleep_data_file(file_path):
    """
    Process a sleep data file (CSV) to extract sleep metrics.
    
    Args:
        file_path (str): Path to the sleep data file
    
    Returns:
        dict: Sleep metrics
    """
    try:
        # Read the sleep data file
        df = pd.read_csv(file_path)
        
        # Extract required metrics
        metrics = {
            'total_sleep_time': df['total_sleep_time'].values[0],
            'rem_time': df['rem_time'].values[0],
            'rem_percentage': df['rem_percentage'].values[0],
            'rem_cycles': df['rem_cycles'].values[0],
            'rem_awakenings': df['rem_awakenings'].values[0]
        }
        
        logger.info(f"Extracted sleep metrics from file: {metrics}")
        return metrics
    
    except Exception as e:
        logger.error(f"Error processing sleep data file {file_path}: {e}")
        return None

def main():
    """Main function to predict mood from sleep data."""
    parser = argparse.ArgumentParser(description="Predict mood based on sleep patterns")
    parser.add_argument(
        "--input", 
        required=True,
        help="Path to input file (EEG file or sleep metrics CSV)"
    )
    parser.add_argument(
        "--rem_model", 
        choices=["logistic_regression", "svm", "random_forest", "xgboost"],
        default="random_forest",
        help="Model to use for REM detection (default: random_forest)"
    )
    parser.add_argument(
        "--mood_model", 
        choices=["logistic_regression", "svm", "random_forest", "xgboost"],
        default="xgboost",
        help="Model to use for mood prediction (default: xgboost)"
    )
    parser.add_argument(
        "--output", 
        help="Path to output file (JSON)"
    )
    args = parser.parse_args()
    
    # Check if input file exists
    input_file = Path(args.input)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    # Determine file type
    file_type = input_file.suffix.lower()
    
    # Process the file
    if file_type in ['.edf']:
        # Process EEG file
        logger.info(f"Processing EEG file: {input_file}")
        
        # Extract features
        eeg_features = process_eeg_file(input_file)
        
        if eeg_features is None:
            return
        
        # Detect REM sleep
        eeg_features_with_rem = detect_rem_sleep(eeg_features, args.rem_model)
        
        if eeg_features_with_rem is None:
            return
        
        # Extract sleep metrics
        sleep_metrics = extract_sleep_metrics(eeg_features_with_rem)
        
    elif file_type in ['.csv']:
        # Process sleep data file
        logger.info(f"Processing sleep data file: {input_file}")
        sleep_metrics = process_sleep_data_file(input_file)
        
    else:
        logger.error(f"Unsupported file type: {file_type}")
        return
    
    if sleep_metrics is None:
        return
    
    # Predict mood
    mood_result = predict_mood(sleep_metrics, args.mood_model)
    
    if mood_result is None:
        return
    
    # Combine results
    result = {
        'sleep_metrics': sleep_metrics,
        'mood_prediction': mood_result
    }
    
    # Save results
    if args.output:
        import json
        
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Saved results to {output_file}")
    
    # Print results
    logger.info("\nResults:")
    logger.info(f"Sleep Metrics:")
    for key, value in sleep_metrics.items():
        logger.info(f"  {key}: {value:.1f}")
    
    logger.info(f"\nMood Prediction:")
    logger.info(f"  Predicted Mood: {'Good' if mood_result['good_mood'] else 'Bad'}")
    logger.info(f"  Confidence: {mood_result['good_mood_probability']:.2f}")
    logger.info(f"  Mood Score (0-10): {mood_result['mood_score']:.1f}")

if __name__ == "__main__":
    main() 