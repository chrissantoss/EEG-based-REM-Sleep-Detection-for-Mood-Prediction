# MSE 446 Project Proposal
## Group 25: Siraj Khanna, Seung-woo Kim, David McGuire, Luca Perrone, Chris Santos

## Project Overview

Our project focuses on EEG-based REM sleep detection to analyze sleep patterns' effect on mood upon waking up. Disrupted REM sleep is strongly linked to negative mood states such as anxiety and irritability, yet current tools for predicting waking mood rely heavily on subjective questionnaires or manual EEG analysis by professionals. By implementing machine learning methods, we aim to enable real-time mood prediction based on sleep quality metrics.

## Application Area

Our project falls within the healthcare application area, specifically neuroscience and sleep function. We will analyze EEG data to detect REM sleep patterns and predict mood upon waking, implementing various machine learning models to establish relationships between sleep quality metrics and emotional states.

## Potential Risks

Several risks have been identified for this project:
- Variability across EEG signals amongst individuals
- Poor-quality or missing mood labels
- Interference from external factors (diet, stress, medication, lifestyle)
- Hardware and signal variability (from device quality and electrode placement)

## Data Sources

We plan to use publicly available EEG datasets from platforms like PhysioNet, including:
1. **Sleep-EDF Database**: Contains EEG recordings with sleep stage annotations
2. **Sleep-Cassette Study**: Additional EEG recordings with sleep stage annotations
3. **Sleep-Mood Dataset**: A synthetic dataset with sleep stages and mood ratings

Raw EEG data requires preprocessing before use in machine learning models. Our pipeline includes:
- Filtering to remove unwanted frequencies
- Segmentation into smaller windows for analysis
- Feature extraction from time and frequency domains
- Normalization to ensure consistency across datasets

## Feature Extraction

Our data will contain the following features:

### Time Domain Features
- Mean, standard deviation, and peak-to-peak amplitude
- Signal integrity and variability over time

### Frequency Domain Features
- Power spectral density
- Peak frequency
- Power distribution across different frequency bands (delta, theta, alpha, beta)

### Connectivity Features
- Synchronization between brain regions
- Phase relationships between signals

## Project Timeline

### February 7
- Finalize dataset selection
- Begin feature extraction
- Establish standardized preprocessing pipeline

### February 14
- Finalize feature extraction and selection
- Conduct exploratory data analysis (EDA)
- Visualize trends and correlations in EEG data

### February 21
- Select and implement baseline models (Logistic Regression, SVM)
- Implement more complex models (Random Forest, XGBoost)
- Begin model tuning

### March 17
- Complete model training and hyperparameter tuning
- Evaluate models using metrics (accuracy, F1 score)
- Perform cross-validation and error analysis

### March 17-25
- Prepare presentation slides
- Refine codebase for submission
- Rehearse presentation

## Model Selection

We will implement and compare several models:

1. **Logistic Regression**: Simple baseline to classify mood as positive or negative
2. **Support Vector Machines**: Using an RBF kernel to capture nonlinear relationships
3. **Random Forest**: Nonparametric approach to handle EEG noise and identify relevant features
4. **XGBoost**: Gradient-boosting algorithm to improve accuracy by combining multiple decision trees

## Project Structure

Our project is organized as follows:

```
eeg-rem-sleep-detection/
├── data/                      # Data directory
│   ├── raw/                   # Raw EEG datasets
│   ├── processed/             # Processed datasets
│   └── features/              # Extracted features
├── notebooks/                 # Jupyter notebooks for exploration
├── src/                       # Source code
│   ├── data/                  # Data processing scripts
│   ├── features/              # Feature extraction
│   ├── models/                # ML models implementation
│   ├── visualization/         # Visualization tools
│   └── utils/                 # Utility functions
├── tests/                     # Unit tests
├── models/                    # Trained models
├── visualizations/            # Generated visualizations
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Key Scripts

- `init_project.py`: Initializes the project structure
- `test_setup.py`: Tests the project setup and dependencies
- `run_pipeline.py`: Runs the entire pipeline with a single command
- `src/data/download_datasets.py`: Downloads datasets from PhysioNet
- `src/data/process_data.py`: Processes raw EEG data
- `src/features/extract_features.py`: Extracts features for model training
- `src/models/train_model.py`: Trains machine learning models
- `src/models/predict_mood.py`: Predicts mood based on sleep patterns
- `src/visualization/visualize_sleep_patterns.py`: Creates visualizations
- `src/utils/run_pipeline.py`: Runs the entire pipeline

## Conclusion

Our EEG-based REM sleep detection project aims to establish a clear relationship between sleep quality and mood upon waking. By implementing machine learning techniques, we hope to provide a more objective and automated approach to mood prediction based on sleep patterns. This could have significant implications for mental health monitoring and sleep disorder treatment. 