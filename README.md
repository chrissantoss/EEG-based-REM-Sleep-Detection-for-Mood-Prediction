# EEG-based REM Sleep Detection for Mood Prediction

This project analyzes EEG data to detect REM sleep patterns and predict mood upon waking. It implements various machine learning models to establish relationships between sleep quality metrics and emotional states.

## Project Overview

Sleep quality, particularly during REM phases, has significant impacts on mood and cognitive function. This project:

1. Processes raw EEG data from public datasets
2. Extracts time and frequency domain features
3. Implements multiple ML models to detect REM sleep patterns
4. Predicts waking mood based on sleep quality metrics
5. Provides visualization tools for sleep pattern analysis

## Data Resources

### Datasets Used

This project uses the following datasets for training and evaluation:

1. **Sleep-EDF Database** (PhysioNet):
   - Contains whole-night polysomnographic sleep recordings
   - Includes EEG, EOG, chin EMG, and event markers
   - [Access on PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/)

2. **Sleep-Cassette Study** (PhysioNet):
   - Contains 153 whole-night PolySomnoGraphic sleep recordings
   - Includes EEG recordings with sleep stage annotations
   - [Access on PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/)

### Accessing the Data

#### Option 1: Download from Source

1. Run the provided data download script:
   ```bash
   python src/data/download_datasets.py
   ```
   This script will download the Sleep-EDF and Sleep-Cassette datasets from PhysioNet.

#### Option 2: Use Pre-packaged Data (Recommended for Quick Start)

The processed data files are already included in the repository:
- `data/processed/sleep_efficiency_research_based.csv`: Pre-processed sleep efficiency data with mood labels

This data is ready to use with the models and doesn't require additional processing.

### Data Structure

#### sleep_efficiency_research_based.csv
- **Records**: 454 sleep sessions
- **Features**: 
  - `total_sleep_time` (minutes)
  - `wake_time` (minutes)
  - `rem_time` (minutes)
  - `light_sleep_time` (minutes)
  - `deep_sleep_time` (minutes)
  - `sleep_efficiency` (percentage)
  - `rem_percentage` (percentage)
  - `good_mood` (binary: 0=negative, 1=positive)

### Processing Raw Data

If you need to process the raw EEG data from scratch:

```bash
# Process the raw EEG data from Sleep-EDF and Sleep-Cassette
python run_pipeline.py pipeline --steps process --datasets sleep-edf,sleep-cassette

# Extract features from processed data
python run_pipeline.py pipeline --steps extract --task all
```

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

4. **Access the Data**:
   See the [Data Resources](#data-resources) section above for details on accessing the datasets required for this project.

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

## Running the Project

### Quick Start

For a quick start with pre-trained models and processed data:

```bash
# Test all mood prediction models and compare their performance
python test_all_models.py

# Analyze sleep features to understand their relationship with mood
python analyze_sleep_features.py

# Evaluate the enhanced mood prediction model
python enhanced_mood_prediction.py
```

### Running the Complete Pipeline

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
python predict_waking_mood.py
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

### Advanced XGBoost Hyperparameter Tuning

For more intensive tuning of the XGBoost model specifically for mood prediction, the project includes a dedicated script:

```bash
python tune_mood_prediction.py
```

This script performs an advanced hyperparameter optimization process:

1. **Baseline Evaluation**: First evaluates a baseline XGBoost model with default parameters
2. **Extended Parameter Search**: Explores a comprehensive parameter space including:
   - Tree parameters: max_depth, min_child_weight, gamma
   - Boosting parameters: learning_rate, n_estimators
   - Sampling parameters: subsample, colsample_bytree, colsample_bylevel
   - Regularization parameters: reg_alpha, reg_lambda
   - Class imbalance handling: scale_pos_weight

3. **Two-Stage Tuning**:
   - Initial RandomizedSearchCV with 50 parameter combinations
   - Fine-tuning of the learning rate with a narrow search around the best value

4. **Results Analysis**:
   - Compares metrics between baseline and tuned models
   - Generates detailed logs of performance improvements
   - Saves both an initially tuned model and a fine-tuned model

The tuning process typically takes several minutes and generates models with metrics approaching or exceeding:
- F1 score: 94-95%
- Accuracy: 92-93%
- ROC AUC: 97-98%

Each tuned model is saved with a timestamp and complete metadata, including:
- The exact hyperparameters used
- Performance metrics on test data
- Percentage improvement over the baseline

Models are saved to:
```
models/mood_prediction/xgboost_tuned_[TIMESTAMP].joblib
models/mood_prediction/xgboost_finetuned_[TIMESTAMP].joblib
```

Tuning results are saved to:
```
models/tuning_results/mood_prediction/xgboost_tuning_results_[TIMESTAMP].json
```

### Robust Model Implementation

The project includes robust versions of both XGBoost and Random Forest models specifically designed to handle outliers, noise, and edge cases in sleep data. These models aim to achieve higher performance metrics (>95% F1 score) and improved reliability in real-world scenarios.

#### Model Features

The robust models incorporate several advanced techniques:

1. **Ensemble Architecture**:
   - XGBoost Robust combines multiple XGBoost models with different hyperparameters using a soft voting ensemble
   - Random Forest Robust uses cross-validated predictions as additional features to improve stability

2. **Advanced Preprocessing**:
   - Uses `RobustScaler` instead of standard scaling to handle outliers
   - Applies specialized feature engineering for sleep metrics

3. **Derived Features**:
   - Sleep continuity metric (ratio of wake time to total sleep time)
   - Sleep depth ratio (deep sleep to light sleep ratio)
   - Recovery ratio (deep sleep to wake time ratio)
   - Composite sleep score weighted by different sleep phases
   - Sleep quality index combining multiple sleep quality factors

4. **Enhanced Robustness**:
   - Models are less sensitive to outliers and measurement errors
   - Improved performance on edge cases and atypical sleep patterns
   - Better generalization to unseen data

#### Running Robust Models

To create and test the robust models:

```bash
# Create the robust models
python robust_mood_models.py

# Test all models including the robust versions
python test_all_models.py
```

This will create:
- `models/mood_prediction/xgboost_robust.joblib`
- `models/mood_prediction/random_forest_robust.joblib`

#### Expected Performance

The robust models aim to achieve:
- F1 score: >95%
- Accuracy: >93%
- ROC AUC: >98%

Most importantly, they maintain high performance even on problematic sleep data with outliers or unusual patterns. They excel in scenarios where:
- Sleep data contains noise or outliers
- Sleep patterns are atypical or irregular
- Limited data is available for training

#### Using Robust Models for Prediction

To use a robust model for mood prediction:

```bash
python predict_waking_mood.py --model xgboost_robust
```

Or integrate directly into your application:

```python
from pathlib import Path
import joblib

# Load the robust model
model_path = Path("models/mood_prediction/xgboost_robust.joblib")
model = joblib.load(model_path)

# Make predictions
sleep_metrics = {
    'total_sleep_time': 420, 
    'sleep_efficiency': 0.85,
    'rem_percentage': 0.25,
    # ... other sleep metrics
}

# Convert to appropriate format
import pandas as pd
X = pd.DataFrame([sleep_metrics])

# Predict
prediction = model.predict(X)
probability = model.predict_proba(X)[:, 1]
```

#### Interpreting Tuning Results

When running `tune_mood_prediction.py`, you'll see detailed logs with the following key information:

```
Baseline model metrics:
  F1 score: 0.9496
  Accuracy: 0.9231
  ROC AUC: 0.9869
```

### Known Issues and Troubleshooting

#### Metadata Issue with Tuned Models

When running `test_all_models.py` after tuning, you may notice warnings like:

```
WARNING - No feature information found for xgboost_tuned_20250331_151820, cannot evaluate safely
```

This occurs because feature metadata is not properly saved during the tuning process. To fix this:

1. **Update Metadata for Tuned Models**:

```python
import joblib
from pathlib import Path

# Adjust these paths to your tuned model
tuned_model_path = "models/mood_prediction/xgboost_tuned_TIMESTAMP.joblib"
metadata_path = "models/mood_prediction/xgboost_tuned_TIMESTAMP_metadata.joblib"

# Load existing metadata
metadata = joblib.load(metadata_path)

# Add feature information
metadata['features'] = [
    'total_sleep_time', 'wake_time', 'rem_time', 'light_sleep_time', 
    'deep_sleep_time', 'sleep_efficiency', 'rem_percentage', 'stress_level',
    'exercise_minutes', 'caffeine_mg', 'alcohol_units', 'sleep_quality_index', 
    'sleep_continuity', 'sleep_depth_ratio', 'stress_exercise_balance', 
    'recovery_ratio', 'composite_sleep_score'
]

# Save updated metadata
joblib.dump(metadata, metadata_path)
```

After updating the metadata, run `test_all_models.py` again to properly evaluate all models.

## Step-by-Step Workflow for Reproduction

Here's a recommended workflow to reproduce key results for your experiments:

1. **Setup the environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Explore the data**:
   ```bash
   # View the sleep efficiency dataset
   python -c "import pandas as pd; print(pd.read_csv('data/processed/sleep_efficiency_research_based.csv').head())"
   ```

3. **Run feature analysis**:
   ```bash
   python analyze_sleep_features.py
   ```

4. **Compare all models**:
   ```bash
   python test_all_models.py
   ```

5. **Run advanced hyperparameter tuning** (optional for improved performance):
   ```bash
   # Run advanced XGBoost hyperparameter tuning for mood prediction
   python tune_mood_prediction.py
   
   # After tuning completes, run the model comparison again to see improvements
   python test_all_models.py
   ```

6. **Generate visualizations for results**:
   ```bash
   python run_pipeline.py visualize --task mood_prediction
   ```

7. **View the best model results**:
   ```bash
   python test_best_model.py
   ```

## Testing

Run the test suite:

```bash
pytest tests/
```

## Expected Results

When running the model evaluation, you should expect to see:

- The best XGBoost model (fine-tuned and refined) achieving the best performance with:
  - F1 score: ~94.9%
  - Accuracy: ~92.3%
  - Precision: ~91.5%
  - Recall: ~97.0%
  - ROC AUC: ~98.5%

- The XGBoost enhanced model also performs well with:
  - F1 score: ~94.2%
  - Accuracy: ~91.2%
  - Precision: ~90.4%
  - Recall: ~97.0%
  - ROC AUC: ~97.7%

**Note**: The names of models in the evaluation results will match the timestamp of when they were created (e.g., `xgboost_finetuned_20250331_152458_refined_refined`).

## Contributors

Group 25: Siraj Khanna, Seung-woo Kim, David McGuire, Luca Perrone, Chris Santos

## License

MIT
