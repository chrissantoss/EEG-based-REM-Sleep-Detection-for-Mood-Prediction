# EEG-based REM Sleep Detection for Mood Prediction

This project analyzes EEG data to detect REM sleep patterns and predict mood upon waking. It implements various machine learning models to establish relationships between sleep quality metrics and emotional states.

## Project Overview

Sleep quality, particularly during REM phases, has significant impacts on mood and cognitive function. This project:

1. Processes raw EEG data from public datasets
2. Extracts time and frequency domain features
3. Implements multiple ML models to detect REM sleep patterns
4. Predicts waking mood based on sleep quality metrics
5. Provides visualization tools for sleep pattern analysis

## Setup Instructions

### Prerequisites

- Python 3.9+
- pip (Python package manager)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/eeg-rem-sleep-detection.git
cd eeg-rem-sleep-detection
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download datasets:
The project uses publicly available EEG datasets. Run the data download script:
```bash
python src/data/download_datasets.py
```

## Project Structure

```
eeg-rem-sleep-detection/
├── data/                      # Data directory
│   ├── raw/                   # Raw EEG datasets
│   ├── processed/             # Processed datasets
│   └── features/              # Extracted features
├── models/                    # Trained models
│   ├── rem_detection/         # REM detection models
│   ├── mood_prediction/       # Mood prediction models
│   └── tuning_results/        # Hyperparameter tuning results
├── notebooks/                 # Jupyter notebooks for exploration and visualization
├── src/                       # Source code
│   ├── data/                  # Data processing scripts
│   ├── features/              # Feature extraction
│   ├── models/                # ML models implementation
│   ├── visualization/         # Visualization tools
│   └── utils/                 # Utility functions
├── tests/                     # Unit tests
├── visualizations/            # Generated visualizations
│   └── tuning_results/        # Hyperparameter tuning visualizations
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Usage

### Running the Pipeline

The project provides a unified command-line interface for running different components:

```bash
python run_pipeline.py [command] [options]
```

Available commands:
- `pipeline`: Run the standard data processing and model training pipeline
- `tune`: Run hyperparameter tuning for models
- `visualize`: Visualize hyperparameter tuning results

### Data Processing

```bash
python run_pipeline.py pipeline --steps process
```

### Feature Extraction

```bash
python run_pipeline.py pipeline --steps extract
```

### Model Training

```bash
python run_pipeline.py pipeline --steps train --model [model_name]
```
Available models: logistic_regression, svm, random_forest, xgboost

### Mood Prediction

```bash
python src/models/predict_mood.py --input [eeg_data_file]
```

### Visualization

```bash
python run_pipeline.py pipeline --steps visualize
```

## Hyperparameter Tuning

The project includes a comprehensive hyperparameter tuning system that helps optimize model performance and tracks results.

### Running Hyperparameter Tuning

```bash
python run_pipeline.py tune [options]
```

Options:
- `--task`: Task to tune models for (`rem_detection`, `mood_prediction`, or `all`)
- `--model`: Model to tune (`logistic_regression`, `svm`, `random_forest`, `xgboost`, or `all`)
- `--n_iter`: Number of parameter settings to sample (default: 50)
- `--cv`: Number of cross-validation folds (default: 5)
- `--scoring`: Scoring metric for hyperparameter tuning (`accuracy`, `precision`, `recall`, `f1`, or `roc_auc`)

Examples:

```bash
# Tune all models for all tasks with default settings
python run_pipeline.py tune

# Tune only the random forest model for REM detection with 100 iterations
python run_pipeline.py tune --task rem_detection --model random_forest --n_iter 100
```

### Visualizing Tuning Results

```bash
python run_pipeline.py visualize [options]
```

Options:
- `--task`: Filter by task name
- `--model`: Filter by model name
- `--output-dir`: Directory to save visualizations

The visualization script generates:
1. Metric comparison charts (before vs. after tuning)
2. Improvement heatmaps
3. Parameter importance visualizations
4. Comprehensive HTML reports

For more details, see the [Hyperparameter Tuning README](models/tuning_results/README.md).

## Testing

Run the test suite:

```bash
pytest tests/
```

## Contributors

Group 25: Siraj Khanna, Seung-woo Kim, David McGuire, Luca Perrone, Chris Santos

## License

MIT
