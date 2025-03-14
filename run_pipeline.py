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
    
    # Check if the pipeline script exists
    pipeline_script = Path("src/utils/run_pipeline.py")
    if not pipeline_script.exists():
        logger.error(f"Pipeline script not found: {pipeline_script}")
        logger.info("Please run init_project.py first to set up the project structure.")
        return 1
    
    # Run the pipeline
    return run_pipeline(args)

if __name__ == "__main__":
    sys.exit(main()) 