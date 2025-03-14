#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script processes raw EEG data for sleep stage analysis.
It includes filtering, segmentation, and normalization steps.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import mne
import pyedflib
from pathlib import Path
from tqdm import tqdm

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
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Ensure directories exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Sleep stage mapping (according to AASM standards)
SLEEP_STAGE_MAP = {
    'W': 0,    # Wake
    'N1': 1,   # NREM 1 (Light sleep)
    'N2': 2,   # NREM 2 (Light sleep)
    'N3': 3,   # NREM 3 (Deep sleep)
    'R': 4,    # REM sleep
    'M': 5,    # Movement
    '?': 6,    # Unknown
}

# Reverse mapping for readability
SLEEP_STAGE_NAMES = {
    0: 'Wake',
    1: 'N1',
    2: 'N2',
    3: 'N3',
    4: 'REM',
    5: 'Movement',
    6: 'Unknown',
}

def read_edf_file(file_path):
    """
    Read an EDF file using MNE.
    
    Args:
        file_path (Path): Path to the EDF file
    
    Returns:
        mne.io.Raw: MNE Raw object containing the EEG data
    """
    try:
        # Read the EDF file
        raw = mne.io.read_raw_edf(file_path, preload=True)
        logger.info(f"Successfully read EDF file: {file_path}")
        return raw
    except Exception as e:
        logger.error(f"Error reading EDF file {file_path}: {e}")
        return None

def read_hypnogram(file_path):
    """
    Read a hypnogram file (sleep stage annotations).
    
    Args:
        file_path (Path): Path to the hypnogram file
    
    Returns:
        pd.DataFrame: DataFrame containing sleep stage annotations
    """
    try:
        # Read the hypnogram file
        if file_path.suffix == '.edf':
            # For EDF hypnograms
            f = pyedflib.EdfReader(str(file_path))
            annotations = f.readAnnotations()
            f.close()
            
            # Convert to DataFrame
            hypnogram = pd.DataFrame({
                'onset': annotations[0],
                'duration': annotations[1],
                'description': annotations[2]
            })
            
            # Extract sleep stages
            hypnogram['sleep_stage'] = hypnogram['description'].apply(
                lambda x: x.split(' ')[0] if ' ' in x else x
            )
            
            # Map to numeric values
            hypnogram['sleep_stage_num'] = hypnogram['sleep_stage'].map(
                lambda x: SLEEP_STAGE_MAP.get(x, 6)  # Default to Unknown
            )
            
        else:
            # For other formats (e.g., CSV)
            hypnogram = pd.read_csv(file_path)
        
        logger.info(f"Successfully read hypnogram file: {file_path}")
        return hypnogram
    
    except Exception as e:
        logger.error(f"Error reading hypnogram file {file_path}: {e}")
        return None

def filter_eeg_data(raw, l_freq=0.5, h_freq=30.0):
    """
    Apply bandpass filter to EEG data.
    
    Args:
        raw (mne.io.Raw): MNE Raw object containing the EEG data
        l_freq (float): Lower frequency bound
        h_freq (float): Upper frequency bound
    
    Returns:
        mne.io.Raw: Filtered MNE Raw object
    """
    try:
        # Apply bandpass filter
        raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq)
        logger.info(f"Applied bandpass filter ({l_freq}-{h_freq} Hz)")
        return raw_filtered
    
    except Exception as e:
        logger.error(f"Error applying filter: {e}")
        return raw

def segment_eeg_data(raw, window_size=30.0, overlap=0.0):
    """
    Segment EEG data into fixed-size windows.
    
    Args:
        raw (mne.io.Raw): MNE Raw object containing the EEG data
        window_size (float): Window size in seconds
        overlap (float): Overlap between windows in seconds
    
    Returns:
        list: List of EEG segments
    """
    try:
        # Get data and sampling frequency
        data, times = raw.get_data(return_times=True)
        sfreq = raw.info['sfreq']
        
        # Calculate window and step sizes in samples
        window_samples = int(window_size * sfreq)
        step_samples = int((window_size - overlap) * sfreq)
        
        # Segment the data
        segments = []
        for start in range(0, data.shape[1] - window_samples + 1, step_samples):
            end = start + window_samples
            segment = data[:, start:end]
            segments.append(segment)
        
        logger.info(f"Segmented data into {len(segments)} windows of {window_size}s with {overlap}s overlap")
        return segments
    
    except Exception as e:
        logger.error(f"Error segmenting data: {e}")
        return []

def extract_features(segments, sfreq):
    """
    Extract time and frequency domain features from EEG segments.
    
    Args:
        segments (list): List of EEG segments
        sfreq (float): Sampling frequency
    
    Returns:
        pd.DataFrame: DataFrame containing extracted features
    """
    from scipy import stats
    from scipy.signal import welch
    
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
            segment_features[f'ptp_ch{ch_idx}'] = np.ptp(channel_data)  # Peak-to-peak amplitude
            segment_features[f'skew_ch{ch_idx}'] = stats.skew(channel_data)
            segment_features[f'kurtosis_ch{ch_idx}'] = stats.kurtosis(channel_data)
            
            # Frequency domain features
            freqs, psd = welch(channel_data, fs=sfreq, nperseg=min(256, len(channel_data)))
            
            # Delta band (0.5-4 Hz)
            delta_idx = np.logical_and(freqs >= 0.5, freqs <= 4)
            segment_features[f'delta_power_ch{ch_idx}'] = np.sum(psd[delta_idx])
            
            # Theta band (4-8 Hz)
            theta_idx = np.logical_and(freqs >= 4, freqs <= 8)
            segment_features[f'theta_power_ch{ch_idx}'] = np.sum(psd[theta_idx])
            
            # Alpha band (8-13 Hz)
            alpha_idx = np.logical_and(freqs >= 8, freqs <= 13)
            segment_features[f'alpha_power_ch{ch_idx}'] = np.sum(psd[alpha_idx])
            
            # Beta band (13-30 Hz)
            beta_idx = np.logical_and(freqs >= 13, freqs <= 30)
            segment_features[f'beta_power_ch{ch_idx}'] = np.sum(psd[beta_idx])
            
            # Relative band powers
            total_power = np.sum(psd)
            if total_power > 0:
                segment_features[f'delta_rel_power_ch{ch_idx}'] = np.sum(psd[delta_idx]) / total_power
                segment_features[f'theta_rel_power_ch{ch_idx}'] = np.sum(psd[theta_idx]) / total_power
                segment_features[f'alpha_rel_power_ch{ch_idx}'] = np.sum(psd[alpha_idx]) / total_power
                segment_features[f'beta_rel_power_ch{ch_idx}'] = np.sum(psd[beta_idx]) / total_power
            
            # Peak frequencies
            if len(psd) > 0:
                segment_features[f'peak_freq_ch{ch_idx}'] = freqs[np.argmax(psd)]
                
                # Peak frequencies in each band
                if np.sum(delta_idx) > 0 and np.sum(psd[delta_idx]) > 0:
                    segment_features[f'peak_delta_freq_ch{ch_idx}'] = freqs[delta_idx][np.argmax(psd[delta_idx])]
                if np.sum(theta_idx) > 0 and np.sum(psd[theta_idx]) > 0:
                    segment_features[f'peak_theta_freq_ch{ch_idx}'] = freqs[theta_idx][np.argmax(psd[theta_idx])]
                if np.sum(alpha_idx) > 0 and np.sum(psd[alpha_idx]) > 0:
                    segment_features[f'peak_alpha_freq_ch{ch_idx}'] = freqs[alpha_idx][np.argmax(psd[alpha_idx])]
                if np.sum(beta_idx) > 0 and np.sum(psd[beta_idx]) > 0:
                    segment_features[f'peak_beta_freq_ch{ch_idx}'] = freqs[beta_idx][np.argmax(psd[beta_idx])]
        
        # Add segment index
        segment_features['segment_idx'] = i
        
        features.append(segment_features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features)
    logger.info(f"Extracted {features_df.shape[1]} features from {len(segments)} segments")
    
    return features_df

def normalize_features(features_df):
    """
    Normalize features to have zero mean and unit variance.
    
    Args:
        features_df (pd.DataFrame): DataFrame containing features
    
    Returns:
        pd.DataFrame: Normalized features
    """
    from sklearn.preprocessing import StandardScaler
    
    # Select numerical columns (exclude segment_idx)
    num_cols = features_df.columns.difference(['segment_idx'])
    
    # Create a scaler
    scaler = StandardScaler()
    
    # Normalize numerical features
    features_df[num_cols] = scaler.fit_transform(features_df[num_cols])
    
    logger.info("Normalized features")
    return features_df

def align_features_with_hypnogram(features_df, hypnogram, window_size=30.0):
    """
    Align extracted features with sleep stage annotations.
    
    Args:
        features_df (pd.DataFrame): DataFrame containing features
        hypnogram (pd.DataFrame): DataFrame containing sleep stage annotations
        window_size (float): Window size in seconds used for segmentation
    
    Returns:
        pd.DataFrame: Features with sleep stage labels
    """
    # Create a copy of the features DataFrame
    labeled_features = features_df.copy()
    
    # Initialize sleep stage column
    labeled_features['sleep_stage'] = 6  # Unknown
    
    # Iterate through hypnogram entries
    for _, row in hypnogram.iterrows():
        onset = row['onset']
        duration = row['duration']
        stage = row['sleep_stage_num']
        
        # Calculate segment indices that correspond to this annotation
        start_segment = int(onset / window_size)
        end_segment = int((onset + duration) / window_size)
        
        # Assign sleep stage to segments
        segment_mask = (labeled_features['segment_idx'] >= start_segment) & (labeled_features['segment_idx'] < end_segment)
        labeled_features.loc[segment_mask, 'sleep_stage'] = stage
    
    # Add sleep stage name for readability
    labeled_features['sleep_stage_name'] = labeled_features['sleep_stage'].map(SLEEP_STAGE_NAMES)
    
    logger.info(f"Aligned features with hypnogram: {labeled_features['sleep_stage'].value_counts().to_dict()}")
    return labeled_features

def process_eeg_file(eeg_file, hypnogram_file, output_dir):
    """
    Process an EEG file and its corresponding hypnogram.
    
    Args:
        eeg_file (Path): Path to the EEG file
        hypnogram_file (Path): Path to the hypnogram file
        output_dir (Path): Directory to save processed data
    
    Returns:
        Path: Path to the processed data file
    """
    # Read EEG data
    raw = read_edf_file(eeg_file)
    if raw is None:
        return None
    
    # Read hypnogram
    hypnogram = read_hypnogram(hypnogram_file)
    if hypnogram is None:
        return None
    
    # Filter EEG data
    raw_filtered = filter_eeg_data(raw)
    
    # Segment EEG data
    segments = segment_eeg_data(raw_filtered)
    
    # Extract features
    features_df = extract_features(segments, raw.info['sfreq'])
    
    # Normalize features
    normalized_features = normalize_features(features_df)
    
    # Align features with hypnogram
    labeled_features = align_features_with_hypnogram(normalized_features, hypnogram)
    
    # Save processed data
    output_file = output_dir / f"{eeg_file.stem}_processed.csv"
    labeled_features.to_csv(output_file, index=False)
    
    logger.info(f"Saved processed data to {output_file}")
    return output_file

def process_sleep_edf_dataset(dataset_dir, output_dir):
    """
    Process the Sleep-EDF dataset.
    
    Args:
        dataset_dir (Path): Directory containing the Sleep-EDF dataset
        output_dir (Path): Directory to save processed data
    
    Returns:
        list: List of paths to processed data files
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all EEG files
    eeg_files = list(dataset_dir.glob("*-PSG.edf"))
    
    processed_files = []
    
    for eeg_file in tqdm(eeg_files, desc="Processing EEG files"):
        # Find corresponding hypnogram file
        hypnogram_file = dataset_dir / f"{eeg_file.stem.replace('-PSG', '')}-Hypnogram.edf"
        
        if not hypnogram_file.exists():
            # Try alternative naming convention
            hypnogram_file = dataset_dir / f"{eeg_file.stem.replace('-PSG', '')}-Hypnogram.edf"
        
        if not hypnogram_file.exists():
            logger.warning(f"Hypnogram file not found for {eeg_file}")
            continue
        
        # Process the EEG file
        processed_file = process_eeg_file(eeg_file, hypnogram_file, output_dir)
        
        if processed_file is not None:
            processed_files.append(processed_file)
    
    logger.info(f"Processed {len(processed_files)} files from Sleep-EDF dataset")
    return processed_files

def process_sleep_mood_dataset(dataset_dir, output_dir):
    """
    Process the synthetic sleep-mood dataset.
    
    Args:
        dataset_dir (Path): Directory containing the sleep-mood dataset
        output_dir (Path): Directory to save processed data
    
    Returns:
        Path: Path to the processed data file
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the dataset file
    dataset_file = dataset_dir / "sleep_mood_dataset.csv"
    
    if not dataset_file.exists():
        logger.error(f"Dataset file not found: {dataset_file}")
        return None
    
    # Read the dataset
    df = pd.read_csv(dataset_file)
    
    # Save as processed data
    output_file = output_dir / "sleep_mood_processed.csv"
    df.to_csv(output_file, index=False)
    
    logger.info(f"Processed sleep-mood dataset and saved to {output_file}")
    return output_file

def main():
    """Main function to process datasets."""
    parser = argparse.ArgumentParser(description="Process EEG sleep datasets")
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        default=["sleep-edf", "sleep-cassette", "sleep-mood"],
        help="Names of datasets to process (default: all)"
    )
    args = parser.parse_args()
    
    logger.info(f"Starting processing of {len(args.datasets)} datasets")
    
    for dataset_name in args.datasets:
        dataset_dir = RAW_DATA_DIR / dataset_name
        output_dir = PROCESSED_DIR / dataset_name
        
        if not dataset_dir.exists():
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            continue
        
        logger.info(f"Processing dataset: {dataset_name}")
        
        if dataset_name in ["sleep-edf", "sleep-cassette"]:
            process_sleep_edf_dataset(dataset_dir, output_dir)
        elif dataset_name == "sleep-mood":
            process_sleep_mood_dataset(dataset_dir, output_dir)
        else:
            logger.warning(f"Unknown dataset: {dataset_name}")
    
    logger.info("All processing completed")

if __name__ == "__main__":
    main() 