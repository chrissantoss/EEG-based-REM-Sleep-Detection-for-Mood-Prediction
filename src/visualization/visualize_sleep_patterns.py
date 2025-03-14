#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script visualizes sleep patterns and their relationship with mood.
It creates various plots to help understand the connection between REM sleep and mood.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE

# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Define the data directory
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = ROOT_DIR / "models"
VISUALIZATION_DIR = ROOT_DIR / "visualizations"

# Ensure directories exist
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

def load_data(dataset_name):
    """
    Load processed data for visualization.
    
    Args:
        dataset_name (str): Name of the dataset
    
    Returns:
        pd.DataFrame: DataFrame containing processed data
    """
    dataset_dir = PROCESSED_DIR / dataset_name
    
    if not dataset_dir.exists():
        logger.error(f"Processed data directory not found: {dataset_dir}")
        return None
    
    # Find all CSV files in the directory
    csv_files = list(dataset_dir.glob("*.csv"))
    
    if not csv_files:
        logger.error(f"No processed data files found in {dataset_dir}")
        return None
    
    # Load and concatenate all files
    dfs = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            
            # Add dataset source column
            df['dataset'] = dataset_name
            df['file'] = file.name
            
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    
    if not dfs:
        logger.error(f"Failed to load any data from {dataset_name}")
        return None
    
    # Concatenate all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined_df)} samples from {dataset_name}")
    
    return combined_df

def load_features(task):
    """
    Load extracted features for visualization.
    
    Args:
        task (str): Task name ('rem_detection' or 'mood_prediction')
    
    Returns:
        dict: Dictionary containing features
    """
    # Determine the features file
    if task == "rem_detection":
        features_file = FEATURES_DIR / "rem_detection_features.joblib"
    elif task == "mood_prediction":
        features_file = FEATURES_DIR / "mood_prediction_features.joblib"
    else:
        logger.error(f"Unknown task: {task}")
        return None
    
    # Check if the file exists
    if not features_file.exists():
        logger.error(f"Features file not found: {features_file}")
        return None
    
    try:
        # Load the features
        features = joblib.load(features_file)
        logger.info(f"Loaded features for {task} from {features_file}")
        return features
    
    except Exception as e:
        logger.error(f"Error loading features from {features_file}: {e}")
        return None

def load_model_results(model_name, task):
    """
    Load model results for visualization.
    
    Args:
        model_name (str): Name of the model
        task (str): Task name ('rem_detection' or 'mood_prediction')
    
    Returns:
        tuple: (model, metadata)
    """
    # Determine model file paths
    model_dir = MODELS_DIR / task
    model_file = model_dir / f"{model_name}.joblib"
    metadata_file = model_dir / f"{model_name}_metadata.joblib"
    
    # Check if files exist
    if not model_file.exists() or not metadata_file.exists():
        logger.error(f"Model or metadata file not found: {model_file}, {metadata_file}")
        return None, None
    
    try:
        # Load model and metadata
        model = joblib.load(model_file)
        metadata = joblib.load(metadata_file)
        
        logger.info(f"Loaded {model_name} model results for {task}")
        return model, metadata
    
    except Exception as e:
        logger.error(f"Error loading model results: {e}")
        return None, None

def plot_sleep_stages_distribution(data, output_file=None):
    """
    Plot the distribution of sleep stages.
    
    Args:
        data (pd.DataFrame): DataFrame containing sleep stage data
        output_file (Path, optional): Path to save the plot
    """
    if 'sleep_stage_name' not in data.columns:
        logger.error("Sleep stage name column not found in data")
        return
    
    # Count sleep stages
    stage_counts = data['sleep_stage_name'].value_counts()
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create bar plot
    ax = sns.barplot(x=stage_counts.index, y=stage_counts.values)
    
    # Add labels and title
    plt.title('Distribution of Sleep Stages', fontsize=16)
    plt.xlabel('Sleep Stage', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add count labels on top of bars
    for i, count in enumerate(stage_counts.values):
        ax.text(i, count + 0.1, str(count), ha='center', fontsize=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file)
        logger.info(f"Saved sleep stages distribution plot to {output_file}")
    else:
        plt.show()
    
    plt.close()

def plot_rem_vs_non_rem_features(features, output_file=None):
    """
    Plot the distribution of features for REM vs. non-REM sleep.
    
    Args:
        features (dict): Dictionary containing features
        output_file (Path, optional): Path to save the plot
    """
    # Extract data
    X_train = features['X_train']
    y_train = features['y_train']
    selected_features = features['selected_features']
    
    # Select top features (to avoid overcrowding the plot)
    top_features = selected_features[:min(6, len(selected_features))]
    
    # Create a DataFrame with features and target
    df = X_train[top_features].copy()
    df['is_rem'] = y_train
    
    # Melt the DataFrame for easier plotting
    melted_df = pd.melt(df, id_vars=['is_rem'], value_vars=top_features,
                        var_name='Feature', value_name='Value')
    
    # Map target to readable labels
    melted_df['Sleep Stage'] = melted_df['is_rem'].map({0: 'Non-REM', 1: 'REM'})
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Create box plot
    ax = sns.boxplot(x='Feature', y='Value', hue='Sleep Stage', data=melted_df)
    
    # Add labels and title
    plt.title('Feature Distribution: REM vs. Non-REM Sleep', fontsize=16)
    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file)
        logger.info(f"Saved REM vs. non-REM features plot to {output_file}")
    else:
        plt.show()
    
    plt.close()

def plot_feature_importance(model, model_name, task, output_file=None):
    """
    Plot feature importance for a trained model.
    
    Args:
        model (object): Trained model
        model_name (str): Name of the model
        task (str): Task name
        output_file (Path, optional): Path to save the plot
    """
    # Load features to get feature names
    features = load_features(task)
    
    if features is None:
        return
    
    selected_features = features['selected_features']
    
    # Extract feature importance
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models
        importances = np.abs(model.coef_[0])
    else:
        logger.error(f"Model {model_name} does not provide feature importance")
        return
    
    # Create DataFrame with feature names and importance
    importance_df = pd.DataFrame({
        'Feature': selected_features[:len(importances)],
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create bar plot
    ax = sns.barplot(x='Importance', y='Feature', data=importance_df)
    
    # Add labels and title
    plt.title(f'Feature Importance for {model_name.replace("_", " ").title()} - {task.replace("_", " ").title()}', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file)
        logger.info(f"Saved feature importance plot to {output_file}")
    else:
        plt.show()
    
    plt.close()

def plot_confusion_matrix(model, X_test, y_test, model_name, task, output_file=None):
    """
    Plot confusion matrix for a trained model.
    
    Args:
        model (object): Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        model_name (str): Name of the model
        task (str): Task name
        output_file (Path, optional): Path to save the plot
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Define class labels
    if task == "rem_detection":
        class_names = ['Non-REM', 'REM']
    else:
        class_names = ['Bad Mood', 'Good Mood']
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    # Add labels and title
    plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}', fontsize=16)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file)
        logger.info(f"Saved confusion matrix plot to {output_file}")
    else:
        plt.show()
    
    plt.close()

def plot_roc_curve(model, X_test, y_test, model_name, task, output_file=None):
    """
    Plot ROC curve for a trained model.
    
    Args:
        model (object): Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        model_name (str): Name of the model
        task (str): Task name
        output_file (Path, optional): Path to save the plot
    """
    # Compute ROC curve and ROC area
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Add labels and title
    plt.title(f'ROC Curve - {model_name.replace("_", " ").title()}', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    
    # Add legend
    plt.legend(loc="lower right")
    
    # Set axis limits
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file)
        logger.info(f"Saved ROC curve plot to {output_file}")
    else:
        plt.show()
    
    plt.close()

def plot_tsne_visualization(features, task, output_file=None):
    """
    Create t-SNE visualization of the feature space.
    
    Args:
        features (dict): Dictionary containing features
        task (str): Task name
        output_file (Path, optional): Path to save the plot
    """
    # Extract data
    X_train = features['X_train']
    y_train = features['y_train']
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_train)
    
    # Create DataFrame for plotting
    tsne_df = pd.DataFrame({
        'x': X_tsne[:, 0],
        'y': X_tsne[:, 1],
        'label': y_train
    })
    
    # Map labels to readable names
    if task == "rem_detection":
        tsne_df['Label'] = tsne_df['label'].map({0: 'Non-REM', 1: 'REM'})
    else:
        tsne_df['Label'] = tsne_df['label'].map({0: 'Bad Mood', 1: 'Good Mood'})
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    ax = sns.scatterplot(x='x', y='y', hue='Label', data=tsne_df, palette='viridis', alpha=0.7)
    
    # Add labels and title
    plt.title(f't-SNE Visualization - {task.replace("_", " ").title()}', fontsize=16)
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file)
        logger.info(f"Saved t-SNE visualization to {output_file}")
    else:
        plt.show()
    
    plt.close()

def plot_sleep_mood_relationship(data, output_file=None):
    """
    Plot the relationship between sleep metrics and mood.
    
    Args:
        data (pd.DataFrame): DataFrame containing sleep and mood data
        output_file (Path, optional): Path to save the plot
    """
    # Check if required columns exist
    required_cols = ['rem_percentage', 'rem_time', 'mood_rating']
    if not all(col in data.columns for col in required_cols):
        logger.error(f"Required columns not found in data: {required_cols}")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot REM percentage vs. mood
    sns.regplot(x='rem_percentage', y='mood_rating', data=data, ax=axes[0])
    axes[0].set_title('REM Percentage vs. Mood Rating', fontsize=14)
    axes[0].set_xlabel('REM Sleep (%)', fontsize=12)
    axes[0].set_ylabel('Mood Rating (1-10)', fontsize=12)
    
    # Plot REM time vs. mood
    sns.regplot(x='rem_time', y='mood_rating', data=data, ax=axes[1])
    axes[1].set_title('REM Time vs. Mood Rating', fontsize=14)
    axes[1].set_xlabel('REM Sleep Time (minutes)', fontsize=12)
    axes[1].set_ylabel('Mood Rating (1-10)', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file)
        logger.info(f"Saved sleep-mood relationship plot to {output_file}")
    else:
        plt.show()
    
    plt.close()

def create_interactive_sleep_visualization(data, output_file=None):
    """
    Create an interactive visualization of sleep patterns and mood.
    
    Args:
        data (pd.DataFrame): DataFrame containing sleep and mood data
        output_file (Path, optional): Path to save the plot
    """
    # Check if required columns exist
    required_cols = ['subject_id', 'night', 'rem_percentage', 'mood_rating']
    if not all(col in data.columns for col in required_cols):
        logger.error(f"Required columns not found in data: {required_cols}")
        return
    
    # Create a 3D scatter plot
    fig = px.scatter_3d(
        data,
        x='rem_percentage',
        y='rem_cycles',
        z='mood_rating',
        color='mood_rating',
        size='total_sleep_time',
        hover_name='subject_id',
        hover_data=['night', 'rem_time', 'sleep_efficiency', 'rem_awakenings'],
        color_continuous_scale='Viridis',
        title='3D Visualization of Sleep Patterns and Mood'
    )
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='REM Sleep (%)',
            yaxis_title='REM Cycles',
            zaxis_title='Mood Rating (1-10)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # Save or show the plot
    if output_file:
        fig.write_html(output_file)
        logger.info(f"Saved interactive sleep visualization to {output_file}")
    else:
        fig.show()

def visualize_rem_detection(model_name="random_forest"):
    """
    Create visualizations for REM sleep detection.
    
    Args:
        model_name (str): Name of the model to visualize
    """
    # Create output directory
    output_dir = VISUALIZATION_DIR / "rem_detection"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load EEG data
    eeg_data = load_data("sleep-edf")
    
    if eeg_data is not None:
        # Plot sleep stages distribution
        plot_sleep_stages_distribution(
            eeg_data,
            output_file=output_dir / "sleep_stages_distribution.png"
        )
    
    # Load features
    features = load_features("rem_detection")
    
    if features is not None:
        # Plot REM vs. non-REM features
        plot_rem_vs_non_rem_features(
            features,
            output_file=output_dir / "rem_vs_non_rem_features.png"
        )
        
        # Plot t-SNE visualization
        plot_tsne_visualization(
            features,
            "rem_detection",
            output_file=output_dir / "tsne_visualization.png"
        )
    
    # Load model results
    model, metadata = load_model_results(model_name, "rem_detection")
    
    if model is not None and features is not None:
        # Plot feature importance
        plot_feature_importance(
            model,
            model_name,
            "rem_detection",
            output_file=output_dir / f"{model_name}_feature_importance.png"
        )
        
        # Plot confusion matrix
        plot_confusion_matrix(
            model,
            features['X_test'],
            features['y_test'],
            model_name,
            "rem_detection",
            output_file=output_dir / f"{model_name}_confusion_matrix.png"
        )
        
        # Plot ROC curve
        plot_roc_curve(
            model,
            features['X_test'],
            features['y_test'],
            model_name,
            "rem_detection",
            output_file=output_dir / f"{model_name}_roc_curve.png"
        )

def visualize_mood_prediction(model_name="xgboost"):
    """
    Create visualizations for mood prediction.
    
    Args:
        model_name (str): Name of the model to visualize
    """
    # Create output directory
    output_dir = VISUALIZATION_DIR / "mood_prediction"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load sleep-mood data
    sleep_mood_data = load_data("sleep-mood")
    
    if sleep_mood_data is not None:
        # Plot sleep-mood relationship
        plot_sleep_mood_relationship(
            sleep_mood_data,
            output_file=output_dir / "sleep_mood_relationship.png"
        )
        
        # Create interactive visualization
        create_interactive_sleep_visualization(
            sleep_mood_data,
            output_file=output_dir / "interactive_sleep_visualization.html"
        )
    
    # Load features
    features = load_features("mood_prediction")
    
    if features is not None:
        # Plot t-SNE visualization
        plot_tsne_visualization(
            features,
            "mood_prediction",
            output_file=output_dir / "tsne_visualization.png"
        )
    
    # Load model results
    model, metadata = load_model_results(model_name, "mood_prediction")
    
    if model is not None and features is not None:
        # Plot feature importance
        plot_feature_importance(
            model,
            model_name,
            "mood_prediction",
            output_file=output_dir / f"{model_name}_feature_importance.png"
        )
        
        # Plot confusion matrix
        plot_confusion_matrix(
            model,
            features['X_test'],
            features['y_test'],
            model_name,
            "mood_prediction",
            output_file=output_dir / f"{model_name}_confusion_matrix.png"
        )
        
        # Plot ROC curve
        plot_roc_curve(
            model,
            features['X_test'],
            features['y_test'],
            model_name,
            "mood_prediction",
            output_file=output_dir / f"{model_name}_roc_curve.png"
        )

def main():
    """Main function to create visualizations."""
    parser = argparse.ArgumentParser(description="Visualize sleep patterns and mood")
    parser.add_argument(
        "--task", 
        choices=["rem_detection", "mood_prediction", "all"],
        default="all",
        help="Visualization task (default: all)"
    )
    parser.add_argument(
        "--model", 
        choices=["logistic_regression", "svm", "random_forest", "xgboost"],
        default=None,
        help="Model to visualize (default: random_forest for REM detection, xgboost for mood prediction)"
    )
    args = parser.parse_args()
    
    logger.info(f"Starting visualization for task: {args.task}")
    
    if args.task in ["rem_detection", "all"]:
        rem_model = args.model if args.model else "random_forest"
        visualize_rem_detection(rem_model)
    
    if args.task in ["mood_prediction", "all"]:
        mood_model = args.model if args.model else "xgboost"
        visualize_mood_prediction(mood_model)
    
    logger.info("Visualization completed")

if __name__ == "__main__":
    main() 