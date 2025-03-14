#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script tests the project setup and verifies that all dependencies are installed correctly.
It checks for the presence of required packages and directories.
"""

import os
import sys
import logging
import importlib
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Define required packages with their import names
REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "xgboost",
    "mne",
    "pyedflib",
    "matplotlib",
    "seaborn",
    "plotly",
    "jupyter",
    "pytest",
    "tqdm",
    "requests",
    "yaml",
    "pooch"
]

# Define required directories
REQUIRED_DIRECTORIES = [
    "data",
    "data/raw",
    "data/processed",
    "data/features",
    "notebooks",
    "src",
    "src/data",
    "src/features",
    "src/models",
    "src/visualization",
    "src/utils",
    "tests",
    "models",
    "visualizations"
]

# Define required files
REQUIRED_FILES = [
    "requirements.txt",
    "README.md",
    "src/data/download_datasets.py",
    "src/data/process_data.py",
    "src/features/extract_features.py",
    "src/models/train_model.py",
    "src/models/predict_mood.py",
    "src/visualization/visualize_sleep_patterns.py",
    "src/utils/run_pipeline.py",
    "tests/test_pipeline.py"
]

def check_packages():
    """
    Check if required packages are installed.
    
    Returns:
        bool: True if all packages are installed, False otherwise
    """
    all_installed = True
    
    logger.info("Checking required packages...")
    
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            logger.error(f"✗ {package} is not installed")
            all_installed = False
    
    return all_installed

def check_directories(base_dir):
    """
    Check if required directories exist.
    
    Args:
        base_dir (Path): Base directory
    
    Returns:
        bool: True if all directories exist, False otherwise
    """
    all_exist = True
    
    logger.info("Checking required directories...")
    
    for directory in REQUIRED_DIRECTORIES:
        dir_path = base_dir / directory
        if dir_path.exists() and dir_path.is_dir():
            logger.info(f"✓ {directory} exists")
        else:
            logger.error(f"✗ {directory} does not exist")
            all_exist = False
    
    return all_exist

def check_files(base_dir):
    """
    Check if required files exist.
    
    Args:
        base_dir (Path): Base directory
    
    Returns:
        bool: True if all files exist, False otherwise
    """
    all_exist = True
    
    logger.info("Checking required files...")
    
    for file in REQUIRED_FILES:
        file_path = base_dir / file
        if file_path.exists() and file_path.is_file():
            logger.info(f"✓ {file} exists")
        else:
            logger.error(f"✗ {file} does not exist")
            all_exist = False
    
    return all_exist

def main():
    """Main function to test the project setup."""
    parser = argparse.ArgumentParser(description="Test the project setup")
    parser.add_argument(
        "--base-dir", 
        type=Path,
        default=Path.cwd(),
        help="Base directory for the project (default: current directory)"
    )
    args = parser.parse_args()
    
    logger.info(f"Testing project setup in {args.base_dir}")
    
    # Check packages
    packages_ok = check_packages()
    
    # Check directories
    directories_ok = check_directories(args.base_dir)
    
    # Check files
    files_ok = check_files(args.base_dir)
    
    # Print summary
    logger.info("\nSetup Test Summary:")
    logger.info(f"Packages: {'OK' if packages_ok else 'FAILED'}")
    logger.info(f"Directories: {'OK' if directories_ok else 'FAILED'}")
    logger.info(f"Files: {'OK' if files_ok else 'FAILED'}")
    
    if packages_ok and directories_ok and files_ok:
        logger.info("\nAll tests passed! The project is set up correctly.")
        return 0
    else:
        logger.error("\nSome tests failed. Please fix the issues and run the test again.")
        
        if not packages_ok:
            logger.info("\nTo install missing packages, run:")
            logger.info("pip install -r requirements.txt")
        
        if not directories_ok or not files_ok:
            logger.info("\nTo create missing directories and files, run:")
            logger.info("python init_project.py")
        
        return 1

if __name__ == "__main__":
    sys.exit(main()) 