#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script initializes the project structure for the EEG-based REM sleep detection project.
It creates the necessary directories and makes the Python scripts executable.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Define the project structure
PROJECT_STRUCTURE = {
    "data": {
        "raw": {},
        "processed": {},
        "features": {}
    },
    "notebooks": {},
    "src": {
        "data": {},
        "features": {},
        "models": {},
        "visualization": {},
        "utils": {}
    },
    "tests": {},
    "models": {
        "rem_detection": {},
        "mood_prediction": {}
    },
    "visualizations": {
        "rem_detection": {},
        "mood_prediction": {}
    }
}

# Define the scripts to make executable
EXECUTABLE_SCRIPTS = [
    "src/data/download_datasets.py",
    "src/data/process_data.py",
    "src/features/extract_features.py",
    "src/models/train_model.py",
    "src/models/predict_mood.py",
    "src/visualization/visualize_sleep_patterns.py",
    "src/utils/run_pipeline.py",
    "init_project.py"
]

def create_directory_structure(base_dir, structure, current_path=None):
    """
    Create the directory structure recursively.
    
    Args:
        base_dir (Path): Base directory
        structure (dict): Directory structure
        current_path (Path, optional): Current path
    """
    if current_path is None:
        current_path = base_dir
    
    for name, substructure in structure.items():
        path = current_path / name
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")
        
        if substructure:
            create_directory_structure(base_dir, substructure, path)

def make_scripts_executable(base_dir, scripts):
    """
    Make the Python scripts executable.
    
    Args:
        base_dir (Path): Base directory
        scripts (list): List of scripts to make executable
    """
    for script in scripts:
        script_path = base_dir / script
        if script_path.exists():
            try:
                # Make the script executable
                os.chmod(script_path, 0o755)
                logger.info(f"Made script executable: {script_path}")
            except Exception as e:
                logger.error(f"Error making script executable: {script_path} - {e}")
        else:
            logger.warning(f"Script not found: {script_path}")

def main():
    """Main function to initialize the project structure."""
    parser = argparse.ArgumentParser(description="Initialize the project structure")
    parser.add_argument(
        "--base-dir", 
        type=Path,
        default=Path.cwd(),
        help="Base directory for the project (default: current directory)"
    )
    args = parser.parse_args()
    
    logger.info(f"Initializing project structure in {args.base_dir}")
    
    # Create the directory structure
    create_directory_structure(args.base_dir, PROJECT_STRUCTURE)
    
    # Make the scripts executable
    make_scripts_executable(args.base_dir, EXECUTABLE_SCRIPTS)
    
    logger.info("Project structure initialized successfully")
    
    # Print next steps
    logger.info("\nNext steps:")
    logger.info("1. Install dependencies: pip install -r requirements.txt")
    logger.info("2. Download datasets: python src/data/download_datasets.py")
    logger.info("3. Process data: python src/data/process_data.py")
    logger.info("4. Extract features: python src/features/extract_features.py")
    logger.info("5. Train models: python src/models/train_model.py")
    logger.info("6. Create visualizations: python src/visualization/visualize_sleep_patterns.py")
    logger.info("\nOr run the entire pipeline: python src/utils/run_pipeline.py")

if __name__ == "__main__":
    main() 