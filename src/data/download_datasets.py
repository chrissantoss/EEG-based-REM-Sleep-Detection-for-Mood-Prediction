#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script downloads publicly available EEG sleep datasets for analysis.
It uses the Pooch library to manage downloads and caching.
"""

import os
import sys
import logging
import argparse
import yaml
import pooch
import requests
from tqdm import tqdm
from pathlib import Path

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

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configurations
DATASETS = {
    "sleep-edf": {
        "description": "Sleep-EDF Database from PhysioNet - Contains EEG recordings with sleep stage annotations",
        "url": "https://physionet.org/content/sleep-edfx/1.0.0/sleep-cassette/",
        "files": [
            "SC4001E0-PSG.edf", "SC4001EC-Hypnogram.edf",
            "SC4002E0-PSG.edf", "SC4002EC-Hypnogram.edf",
            "SC4011E0-PSG.edf", "SC4011EH-Hypnogram.edf",
            "SC4012E0-PSG.edf", "SC4012EC-Hypnogram.edf"
        ],
        "method": "physionet"
    },
    "sleep-cassette": {
        "description": "Sleep Cassette Study from PhysioNet - Contains EEG recordings with sleep stage annotations",
        "url": "https://physionet.org/content/sleep-edfx/1.0.0/sleep-telemetry/",
        "files": [
            "ST7011J0-PSG.edf", "ST7011JP-Hypnogram.edf",
            "ST7022J0-PSG.edf", "ST7022JM-Hypnogram.edf"
        ],
        "method": "physionet"
    },
    "sleep-mood": {
        "description": "Simulated dataset with sleep stages and mood ratings",
        "url": None,
        "files": ["sleep_mood_dataset.csv"],
        "method": "generate"
    }
}

def download_physionet_file(url, filename, target_dir):
    """
    Download a file from PhysioNet.
    
    Args:
        url (str): Base URL for the dataset
        filename (str): Name of the file to download
        target_dir (Path): Directory to save the file
    
    Returns:
        Path: Path to the downloaded file
    """
    file_url = f"{url.rstrip('/')}/{filename}"
    target_path = target_dir / filename
    
    # Skip if file already exists
    if target_path.exists():
        logger.info(f"File {filename} already exists. Skipping download.")
        return target_path
    
    logger.info(f"Downloading {filename} from {file_url}")
    
    try:
        response = requests.get(file_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with open(target_path, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
                
        return target_path
    
    except Exception as e:
        logger.error(f"Error downloading {filename}: {e}")
        if target_path.exists():
            target_path.unlink()  # Remove partially downloaded file
        return None

def generate_synthetic_dataset(target_dir):
    """
    Generate a synthetic dataset with sleep stages and mood ratings.
    This is used for testing when real datasets are not available.
    
    Args:
        target_dir (Path): Directory to save the generated dataset
    
    Returns:
        Path: Path to the generated dataset file
    """
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    logger.info("Generating synthetic sleep-mood dataset")
    
    # File path
    file_path = target_dir / "sleep_mood_dataset.csv"
    
    # Skip if file already exists
    if file_path.exists():
        logger.info("Synthetic dataset already exists. Skipping generation.")
        return file_path
    
    # Generate synthetic data
    np.random.seed(42)  # For reproducibility
    
    # Number of subjects and nights
    n_subjects = 20
    n_nights = 5
    
    # Lists to store data
    data = []
    
    for subject_id in range(1, n_subjects + 1):
        for night in range(1, n_nights + 1):
            # Generate sleep architecture
            total_sleep_time = np.random.normal(480, 60)  # in minutes
            
            # Sleep stages in minutes
            wake_time = max(0, np.random.normal(30, 15))
            rem_time = max(0, np.random.normal(90, 20))
            light_sleep_time = max(0, np.random.normal(240, 40))
            deep_sleep_time = max(0, total_sleep_time - wake_time - rem_time - light_sleep_time)
            
            # Normalize to ensure total adds up to total_sleep_time
            total = wake_time + rem_time + light_sleep_time + deep_sleep_time
            wake_time = wake_time / total * total_sleep_time
            rem_time = rem_time / total * total_sleep_time
            light_sleep_time = light_sleep_time / total * total_sleep_time
            deep_sleep_time = deep_sleep_time / total * total_sleep_time
            
            # Sleep quality metrics
            sleep_efficiency = (total_sleep_time - wake_time) / total_sleep_time * 100
            rem_percentage = rem_time / total_sleep_time * 100
            
            # Number of REM cycles (typically 4-5 per night)
            rem_cycles = max(1, int(np.random.normal(4.5, 1)))
            
            # REM fragmentation (number of awakenings during REM)
            rem_awakenings = max(0, int(np.random.normal(2, 1.5)))
            
            # Mood ratings (1-10 scale)
            # Mood is influenced by REM sleep quality
            base_mood = np.random.normal(7, 1.5)
            rem_quality_factor = (rem_percentage / 20) * (1 - rem_awakenings / 5)
            mood_rating = max(1, min(10, base_mood + rem_quality_factor))
            
            # Anxiety rating (1-10 scale)
            # Anxiety is negatively correlated with deep sleep
            base_anxiety = np.random.normal(4, 1.5)
            deep_sleep_factor = deep_sleep_time / 120
            anxiety_rating = max(1, min(10, base_anxiety - deep_sleep_factor))
            
            # Date
            date = datetime(2023, 1, 1) + timedelta(days=(subject_id-1)*n_nights + night-1)
            
            # Add row to data
            data.append({
                'subject_id': f'S{subject_id:03d}',
                'night': night,
                'date': date.strftime('%Y-%m-%d'),
                'total_sleep_time': round(total_sleep_time, 1),
                'wake_time': round(wake_time, 1),
                'rem_time': round(rem_time, 1),
                'light_sleep_time': round(light_sleep_time, 1),
                'deep_sleep_time': round(deep_sleep_time, 1),
                'sleep_efficiency': round(sleep_efficiency, 1),
                'rem_percentage': round(rem_percentage, 1),
                'rem_cycles': rem_cycles,
                'rem_awakenings': rem_awakenings,
                'mood_rating': round(mood_rating, 1),
                'anxiety_rating': round(anxiety_rating, 1),
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    
    logger.info(f"Synthetic dataset generated and saved to {file_path}")
    return file_path

def download_dataset(dataset_name, target_dir=None):
    """
    Download a specific dataset.
    
    Args:
        dataset_name (str): Name of the dataset to download
        target_dir (Path, optional): Directory to save the dataset
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    if dataset_name not in DATASETS:
        logger.error(f"Dataset '{dataset_name}' not found in available datasets")
        return False
    
    dataset = DATASETS[dataset_name]
    
    if target_dir is None:
        target_dir = RAW_DATA_DIR / dataset_name
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading dataset: {dataset_name} - {dataset['description']}")
    
    if dataset["method"] == "physionet":
        for filename in dataset["files"]:
            download_physionet_file(dataset["url"], filename, target_dir)
    elif dataset["method"] == "generate":
        generate_synthetic_dataset(target_dir)
    else:
        logger.error(f"Unknown download method: {dataset['method']}")
        return False
    
    logger.info(f"Dataset {dataset_name} downloaded successfully")
    return True

def main():
    """Main function to download datasets."""
    parser = argparse.ArgumentParser(description="Download EEG sleep datasets")
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        default=list(DATASETS.keys()),
        help="Names of datasets to download (default: all)"
    )
    args = parser.parse_args()
    
    logger.info(f"Starting download of {len(args.datasets)} datasets")
    
    for dataset_name in args.datasets:
        download_dataset(dataset_name)
    
    logger.info("All downloads completed")

if __name__ == "__main__":
    main() 