# Trained Models for EEG-based REM Sleep Detection

This directory contains trained machine learning models for REM sleep detection and mood prediction.

## Model Organization

The models are organized into subdirectories based on their task:

- `rem_detection/`: Models for detecting REM sleep stages from EEG data
- `mood_prediction/`: Models for predicting mood based on sleep metrics

Each model is saved as a `.joblib` file, along with a corresponding metadata file that contains information about the model's performance and hyperparameters.

## Available Models

### REM Detection Models

- `logistic_regression.joblib`: Logistic Regression model for REM sleep detection
- `svm.joblib`: Support Vector Machine model for REM sleep detection
- `random_forest.joblib`: Random Forest model for REM sleep detection
- `xgboost.joblib`: XGBoost model for REM sleep detection

### Mood Prediction Models

- `logistic_regression.joblib`: Logistic Regression model for mood prediction
- `svm.joblib`: Support Vector Machine model for mood prediction
- `random_forest.joblib`: Random Forest model for mood prediction
- `xgboost.joblib`: XGBoost model for mood prediction

## Model Metadata

Each model has a corresponding metadata file (e.g., `random_forest_metadata.joblib`) that contains:

- Model name and description
- Task (REM detection or mood prediction)
- Performance metrics (accuracy, precision, recall, F1 score, ROC AUC)
- Confusion matrix
- Hyperparameters used for training

## Using the Models

To load and use a trained model:

```python
import joblib

# Load the model
model = joblib.load('models/rem_detection/random_forest.joblib')

# Load the metadata
metadata = joblib.load('models/rem_detection/random_forest_metadata.joblib')

# Print model information
print(f"Model: {metadata['model_name']}")
print(f"Description: {metadata['description']}")
print(f"Accuracy: {metadata['metrics']['accuracy']:.4f}")

# Make predictions
predictions = model.predict(X)
probabilities = model.predict_proba(X)
```

## Training New Models

To train new models, use the `train_model.py` script in the `src/models` directory:

```bash
python src/models/train_model.py --task rem_detection --model random_forest
```

This will train a Random Forest model for REM sleep detection and save it to this directory. 