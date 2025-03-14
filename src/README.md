# Source Code for EEG-based REM Sleep Detection

This directory contains the source code for the EEG-based REM sleep detection project.

## Code Organization

The code is organized into subdirectories:

- `data/`: Scripts for downloading and processing data
- `features/`: Scripts for feature extraction
- `models/`: Scripts for model training and prediction
- `visualization/`: Scripts for creating visualizations
- `utils/`: Utility scripts and helper functions

## Key Scripts

### Data Processing

- `data/download_datasets.py`: Downloads datasets from PhysioNet and generates synthetic data
- `data/process_data.py`: Processes raw EEG data, including filtering, segmentation, and feature extraction

### Feature Extraction

- `features/extract_features.py`: Extracts features from processed data for model training

### Model Training and Prediction

- `models/train_model.py`: Trains machine learning models for REM detection and mood prediction
- `models/predict_mood.py`: Predicts mood based on sleep patterns using trained models

### Visualization

- `visualization/visualize_sleep_patterns.py`: Creates visualizations of sleep patterns and model results

### Utilities

- `utils/run_pipeline.py`: Runs the entire pipeline from data download to visualization

## Running the Code

Each script can be run independently, but the recommended way is to use the pipeline script:

```bash
python src/utils/run_pipeline.py
```

This will run the entire pipeline from data download to visualization.

To run specific steps:

```bash
python src/utils/run_pipeline.py --steps download,process,extract,train,visualize
```

To run with specific datasets, tasks, or models:

```bash
python src/utils/run_pipeline.py --datasets sleep-edf,sleep-mood --task rem_detection --model random_forest
```

## Adding New Code

When adding new code to the project:

1. Follow the existing directory structure
2. Add appropriate documentation and comments
3. Include error handling and logging
4. Add tests in the `tests/` directory
5. Update the README files as needed 