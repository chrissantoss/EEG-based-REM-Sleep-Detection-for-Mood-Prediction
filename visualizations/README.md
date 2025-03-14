# Visualizations for EEG-based REM Sleep Detection

This directory contains visualizations generated from the EEG sleep data and model results.

## Visualization Organization

The visualizations are organized into subdirectories based on their task:

- `rem_detection/`: Visualizations related to REM sleep detection
- `mood_prediction/`: Visualizations related to mood prediction

## Available Visualizations

### REM Detection Visualizations

- `sleep_stages_distribution.png`: Distribution of sleep stages in the dataset
- `rem_vs_non_rem_features.png`: Comparison of feature distributions for REM vs. non-REM sleep
- `tsne_visualization.png`: t-SNE visualization of the feature space
- `{model_name}_feature_importance.png`: Feature importance for each model
- `{model_name}_confusion_matrix.png`: Confusion matrix for each model
- `{model_name}_roc_curve.png`: ROC curve for each model

### Mood Prediction Visualizations

- `sleep_mood_relationship.png`: Relationship between sleep metrics and mood
- `interactive_sleep_visualization.html`: Interactive 3D visualization of sleep patterns and mood
- `tsne_visualization.png`: t-SNE visualization of the feature space
- `{model_name}_feature_importance.png`: Feature importance for each model
- `{model_name}_confusion_matrix.png`: Confusion matrix for each model
- `{model_name}_roc_curve.png`: ROC curve for each model

## Generating Visualizations

To generate visualizations, use the `visualize_sleep_patterns.py` script in the `src/visualization` directory:

```bash
python src/visualization/visualize_sleep_patterns.py --task all
```

This will generate all visualizations and save them to this directory.

To generate visualizations for a specific task and model:

```bash
python src/visualization/visualize_sleep_patterns.py --task rem_detection --model random_forest
```

## Viewing Interactive Visualizations

Some visualizations are interactive HTML files. To view these:

1. Open the HTML file in a web browser
2. Use mouse controls to rotate, zoom, and explore the visualization

## Adding New Visualizations

When adding new visualizations to the project:

1. Update the visualization script in `src/visualization/`
2. Add the new visualization to this README
3. Include a brief description of what the visualization shows 