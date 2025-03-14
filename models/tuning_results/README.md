# Hyperparameter Tuning Results

This directory contains the results of hyperparameter tuning experiments for various machine learning models. The tuning process helps optimize model performance by finding the best combination of hyperparameters.

## Directory Structure

The tuning results are organized by task:

```
tuning_results/
├── rem_detection/           # Results for REM sleep detection models
│   └── *.json               # JSON files containing tuning results
├── mood_prediction/         # Results for mood prediction models
│   └── *.json               # JSON files containing tuning results
└── README.md                # This file
```

## Result File Format

Each JSON file contains detailed information about a hyperparameter tuning run, including:

- Model name and task
- Performance metrics before and after tuning
- Best hyperparameters found
- Improvement metrics
- Cross-validation results
- Tuning time and configuration

## Using the Hyperparameter Tuning System

### Running Hyperparameter Tuning

To run hyperparameter tuning, use the following command:

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

# Tune XGBoost for mood prediction using ROC AUC as the scoring metric
python run_pipeline.py tune --task mood_prediction --model xgboost --scoring roc_auc
```

### Visualizing Tuning Results

To visualize the tuning results, use the following command:

```bash
python run_pipeline.py visualize [options]
```

Options:
- `--task`: Filter by task name
- `--model`: Filter by model name
- `--output-dir`: Directory to save visualizations

Examples:

```bash
# Visualize all tuning results
python run_pipeline.py visualize

# Visualize only results for the SVM model
python run_pipeline.py visualize --model svm

# Visualize only results for mood prediction
python run_pipeline.py visualize --task mood_prediction
```

## Interpreting Results

The visualization script generates several types of visualizations:

1. **Metric Comparisons**: Bar charts comparing performance metrics before and after tuning
2. **Improvement Heatmap**: Heatmap showing the percentage improvement for each model and task
3. **Parameter Importance**: Visualizations of the best parameters for each model and task
4. **HTML Report**: A comprehensive report summarizing all tuning results

These visualizations are saved in the `visualizations/tuning_results` directory.

## Best Practices

- Start with a small number of iterations (e.g., 20-30) to get quick results, then increase for more thorough tuning
- Use cross-validation to ensure robust results (5-10 folds recommended)
- Consider different scoring metrics based on your specific needs (e.g., F1 for imbalanced classes)
- Compare multiple models to find the best one for your task
- Save and track all tuning results to build knowledge over time

## Troubleshooting

If you encounter issues:

1. Check the log output for error messages
2. Ensure that the required data files exist
3. Verify that the model and task names are correct
4. Try reducing the number of iterations or cross-validation folds if the process is too slow
5. Check for sufficient disk space for saving results and visualizations 