#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script runs the entire EEG-based REM sleep detection pipeline.
It downloads data, processes it, extracts features, trains models, and creates visualizations.
"""

import os
import sys
import logging
import argparse
import time
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import project modules
from data.download_datasets import download_dataset
from data.process_data import main as process_data_main
from features.extract_features import main as extract_features_main
from models.train_model import main as train_model_main
from visualization.visualize_sleep_patterns import main as visualize_main

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def run_data_download(args):
    """
    Run the data download step.
    
    Args:
        args (argparse.Namespace): Command-line arguments
    """
    logger.info("Step 1: Downloading datasets")
    
    # Download datasets
    datasets = args.datasets.split(',') if args.datasets else ["sleep-edf", "sleep-cassette", "sleep-mood"]
    
    for dataset_name in datasets:
        download_dataset(dataset_name)
    
    logger.info("Data download completed")

def run_data_processing(args):
    """
    Run the data processing step.
    
    Args:
        args (argparse.Namespace): Command-line arguments
    """
    logger.info("Step 2: Processing datasets")
    
    # Process datasets
    sys.argv = [sys.argv[0]]
    if args.datasets:
        sys.argv.extend(["--datasets"] + args.datasets.split(','))
    
    process_data_main()
    
    logger.info("Data processing completed")

def run_feature_extraction(args):
    """
    Run the feature extraction step.
    
    Args:
        args (argparse.Namespace): Command-line arguments
    """
    logger.info("Step 3: Extracting features")
    
    # Extract features
    sys.argv = [sys.argv[0]]
    if args.task:
        sys.argv.extend(["--task", args.task])
    
    extract_features_main()
    
    logger.info("Feature extraction completed")

def run_model_training(args):
    """
    Run the model training step.
    
    Args:
        args (argparse.Namespace): Command-line arguments
    """
    logger.info("Step 4: Training models")
    
    # Train models
    sys.argv = [sys.argv[0]]
    if args.task:
        sys.argv.extend(["--task", args.task])
    if args.model:
        sys.argv.extend(["--model", args.model])
    
    train_model_main()
    
    logger.info("Model training completed")

def run_visualization(args):
    """
    Run the visualization step.
    
    Args:
        args (argparse.Namespace): Command-line arguments
    """
    logger.info("Step 5: Creating visualizations")
    
    # Create visualizations
    sys.argv = [sys.argv[0]]
    if args.task:
        sys.argv.extend(["--task", args.task])
    if args.model:
        sys.argv.extend(["--model", args.model])
    
    visualize_main()
    
    logger.info("Visualization completed")

def run_pipeline(args):
    """
    Run the entire pipeline.
    
    Args:
        args (argparse.Namespace): Command-line arguments
    """
    logger.info("Starting the EEG-based REM sleep detection pipeline")
    
    # Record start time
    start_time = time.time()
    
    # Run each step
    if args.steps in ["all", "download"]:
        run_data_download(args)
    
    if args.steps in ["all", "process"]:
        run_data_processing(args)
    
    if args.steps in ["all", "extract"]:
        run_feature_extraction(args)
    
    if args.steps in ["all", "train"]:
        run_model_training(args)
    
    if args.steps in ["all", "visualize"]:
        run_visualization(args)
    
    # Record end time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")

def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description="Run the EEG-based REM sleep detection pipeline")
    parser.add_argument(
        "--steps", 
        choices=["all", "download", "process", "extract", "train", "visualize"],
        default="all",
        help="Pipeline steps to run (default: all)"
    )
    parser.add_argument(
        "--datasets", 
        help="Comma-separated list of datasets to use"
    )
    parser.add_argument(
        "--task", 
        choices=["rem_detection", "mood_prediction", "all"],
        help="Task to perform"
    )
    parser.add_argument(
        "--model", 
        choices=["logistic_regression", "svm", "random_forest", "xgboost", "all"],
        help="Model to use"
    )
    args = parser.parse_args()
    
    run_pipeline(args)

if __name__ == "__main__":
    main() 