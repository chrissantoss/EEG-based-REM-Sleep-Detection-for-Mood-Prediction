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
├── notebooks/                 # Jupyter notebooks for exploration and visualization
├── src/                       # Source code
│   ├── data/                  # Data processing scripts
│   ├── features/              # Feature extraction
│   ├── models/                # ML models implementation
│   ├── visualization/         # Visualization tools
│   └── utils/                 # Utility functions
├── tests/                     # Unit tests
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Usage

### Data Processing

```bash
python src/data/process_data.py
```

### Feature Extraction

```bash
python src/features/extract_features.py
```

### Model Training

```bash
python src/models/train_model.py --model [model_name]
```
Available models: logistic_regression, svm, random_forest, xgboost

### Mood Prediction

```bash
python src/models/predict_mood.py --input [eeg_data_file]
```

### Visualization

```bash
python src/visualization/visualize_sleep_patterns.py
```

## Testing

Run the test suite:

```bash
pytest tests/
```

## Contributors

Group 25: Siraj Khanna, Seung-woo Kim, David McGuire, Luca Perrone, Chris Santos

## License

MIT # EEG-based-REM-Sleep-Detection-for-Mood-Prediction
