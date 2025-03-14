#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script runs the entire EEG-based REM sleep detection pipeline.
It is a convenience wrapper around the run_pipeline.py script in the src/utils directory.
"""

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def run_pipeline(args):
    """
    Run the pipeline script.
    
    Args:
        args (argparse.Namespace): Command-line arguments
    
    Returns:
        int: Exit code
    """
    # Build the command
    cmd = [sys.executable, "src/utils/run_pipeline.py"]
    
    # Add arguments
    if args.steps:
        cmd.extend(["--steps", args.steps])
    
    if args.datasets:
        cmd.extend(["--datasets", args.datasets])
    
    if args.task:
        cmd.extend(["--task", args.task])
    
    if args.model:
        cmd.extend(["--model", args.model])
    
    # Run the command
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(cmd, check=True)
        return process.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Pipeline failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        return 1

def run_hyperparameter_tuning(args):
    """
    Run the hyperparameter tuning script.
    
    Args:
        args (argparse.Namespace): Command-line arguments
    
    Returns:
        int: Exit code
    """
    # Build the command
    cmd = [sys.executable, "src/models/hyperparameter_tuning.py"]
    
    # Add arguments
    if args.task:
        cmd.extend(["--task", args.task])
    
    if args.model:
        cmd.extend(["--model", args.model])
    
    if args.n_iter:
        cmd.extend(["--n_iter", str(args.n_iter)])
    
    if args.cv:
        cmd.extend(["--cv", str(args.cv)])
    
    if args.scoring:
        cmd.extend(["--scoring", args.scoring])
    
    # Run the command
    logger.info(f"Running hyperparameter tuning command: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(cmd, check=True)
        return process.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Hyperparameter tuning failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        logger.error(f"Error running hyperparameter tuning: {e}")
        return 1

def visualize_tuning_results(args):
    """
    Run the visualization script for tuning results.
    
    Args:
        args (argparse.Namespace): Command-line arguments
    
    Returns:
        int: Exit code
    """
    # Build the command
    cmd = [sys.executable, "src/visualization/visualize_tuning_results.py"]
    
    # Add arguments
    if args.task:
        cmd.extend(["--task", args.task])
    
    if args.model:
        cmd.extend(["--model", args.model])
    
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    
    # Run the command
    logger.info(f"Running visualization command: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(cmd, check=True)
        return process.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Visualization failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        logger.error(f"Error running visualization: {e}")
        return 1

def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description="Run the EEG-based REM sleep detection pipeline")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run the standard pipeline")
    pipeline_parser.add_argument(
        "--steps", 
        choices=["all", "download", "process", "extract", "train", "visualize"],
        default="all",
        help="Pipeline steps to run (default: all)"
    )
    pipeline_parser.add_argument(
        "--datasets", 
        help="Comma-separated list of datasets to use"
    )
    pipeline_parser.add_argument(
        "--task", 
        choices=["rem_detection", "mood_prediction", "all"],
        help="Task to perform"
    )
    pipeline_parser.add_argument(
        "--model", 
        choices=["logistic_regression", "svm", "random_forest", "xgboost", "all"],
        help="Model to use"
    )
    
    # Hyperparameter tuning command
    tune_parser = subparsers.add_parser("tune", help="Run hyperparameter tuning")
    tune_parser.add_argument(
        "--task", 
        choices=["rem_detection", "mood_prediction", "all"],
        default="all",
        help="Task to tune models for (default: all)"
    )
    tune_parser.add_argument(
        "--model", 
        choices=["logistic_regression", "svm", "random_forest", "xgboost", "all"],
        default="all",
        help="Model to tune (default: all)"
    )
    tune_parser.add_argument(
        "--n_iter", 
        type=int,
        default=50,
        help="Number of parameter settings to sample (default: 50)"
    )
    tune_parser.add_argument(
        "--cv", 
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    tune_parser.add_argument(
        "--scoring", 
        choices=["accuracy", "precision", "recall", "f1", "roc_auc"],
        default="f1",
        help="Scoring metric for hyperparameter tuning (default: f1)"
    )
    
    # Visualization command
    viz_parser = subparsers.add_parser("visualize", help="Visualize tuning results")
    viz_parser.add_argument(
        "--task", 
        help="Filter by task name"
    )
    viz_parser.add_argument(
        "--model", 
        help="Filter by model name"
    )
    viz_parser.add_argument(
        "--output-dir",
        help="Directory to save visualizations"
    )
    
    args = parser.parse_args()
    
    # Default to pipeline if no command is specified
    if not args.command:
        args.command = "pipeline"
        
    # Check if the pipeline script exists
    pipeline_script = Path("src/utils/run_pipeline.py")
    if not pipeline_script.exists():
        logger.error(f"Pipeline script not found: {pipeline_script}")
        logger.info("Please run init_project.py first to set up the project structure.")
        return 1
    
    # Run the appropriate command
    if args.command == "pipeline":
        return run_pipeline(args)
    elif args.command == "tune":
        return run_hyperparameter_tuning(args)
    elif args.command == "visualize":
        return visualize_tuning_results(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 