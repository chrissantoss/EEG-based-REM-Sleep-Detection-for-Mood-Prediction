#!/usr/bin/env python
"""
Train and evaluate all baseline models for mood prediction.

This script trains Logistic Regression, SVM, Random Forest, and XGBoost models
for mood prediction based on sleep features. It performs hyperparameter tuning
for each model and evaluates their performance on a test set.

Usage:
    python train_all_baseline_models.py [--output_dir OUTPUT_DIR] [--seed SEED]
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train and evaluate baseline models for mood prediction')
    parser.add_argument('--output_dir', type=str, default='results/baseline_models',
                        help='Directory to save models and results')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    return parser.parse_args()

def load_features(feature_path='data/features/mood_prediction_features.joblib'):
    """Load features from the specified path.
    
    Args:
        feature_path: Path to the features file
        
    Returns:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
    """
    logger.info(f'Loading features from {feature_path}')
    try:
        features = joblib.load(feature_path)
        
        X = features['features']
        y = features['labels']
        
        # Split data into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.seed, stratify=y
        )
        
        logger.info(f'Training set: {X_train.shape}, Test set: {X_test.shape}')
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f'Error loading features: {e}')
        sys.exit(1)

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        best_model: Trained model with best parameters
        best_params: Best parameters found
        best_score: Best cross-validation score
    """
    logger.info('\n' + '=' * 40)
    logger.info('Training Logistic Regression model')
    logger.info('=' * 40)
    
    # Define parameter grid
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.1, 0.5, 1.0, 5.0],
        'solver': ['liblinear'],
        'class_weight': [None, 'balanced']
    }
    
    # Create model
    model = LogisticRegression(random_state=args.seed, max_iter=1000)
    
    # Grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logger.info(f'Best parameters: {best_params}')
    logger.info(f'Best cross-validation F1 score: {best_score:.4f}')
    
    return best_model, best_params, best_score

def train_svm(X_train, y_train):
    """Train SVM model with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        best_model: Trained model with best parameters
        best_params: Best parameters found
        best_score: Best cross-validation score
    """
    logger.info('\n' + '=' * 40)
    logger.info('Training SVM model')
    logger.info('=' * 40)
    
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1.0, 10.0, 100.0],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf'],
        'class_weight': [None, 'balanced']
    }
    
    # Create model
    model = SVC(probability=True, random_state=args.seed)
    
    # Grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logger.info(f'Best parameters: {best_params}')
    logger.info(f'Best cross-validation F1 score: {best_score:.4f}')
    
    return best_model, best_params, best_score

def train_random_forest(X_train, y_train):
    """Train Random Forest model with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        best_model: Trained model with best parameters
        best_params: Best parameters found
        best_score: Best cross-validation score
    """
    logger.info('\n' + '=' * 40)
    logger.info('Training Random Forest model')
    logger.info('=' * 40)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'class_weight': [None, 'balanced']
    }
    
    # Create model
    model = RandomForestClassifier(random_state=args.seed)
    
    # Grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logger.info(f'Best parameters: {best_params}')
    logger.info(f'Best cross-validation F1 score: {best_score:.4f}')
    
    return best_model, best_params, best_score

def train_xgboost(X_train, y_train):
    """Train XGBoost model with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        best_model: Trained model with best parameters
        best_params: Best parameters found
        best_score: Best cross-validation score
    """
    logger.info('\n' + '=' * 40)
    logger.info('Training XGBoost model')
    logger.info('=' * 40)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Create model
    model = XGBClassifier(random_state=args.seed, use_label_encoder=False, eval_metric='logloss')
    
    # Grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logger.info(f'Best parameters: {best_params}')
    logger.info(f'Best cross-validation F1 score: {best_score:.4f}')
    
    return best_model, best_params, best_score

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    logger.info(f'Evaluating {model_name} model')
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Log metrics
    logger.info('Model evaluation metrics:')
    logger.info(f'  Accuracy:  {accuracy:.4f}')
    logger.info(f'  Precision: {precision:.4f}')
    logger.info(f'  Recall:    {recall:.4f}')
    logger.info(f'  F1 Score:  {f1:.4f}')
    logger.info(f'  ROC AUC:   {roc_auc:.4f}')
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Store metrics in a dictionary
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist()
    }
    
    return metrics

def save_model_and_metrics(model, metrics, params, cv_score, model_name):
    """Save model, metrics, and parameters to disk.
    
    Args:
        model: Trained model
        metrics: Evaluation metrics
        params: Model parameters
        cv_score: Cross-validation score
        model_name: Name of the model
    """
    # Create output directories if they don't exist
    model_dir = os.path.join(args.output_dir, 'models')
    results_dir = os.path.join(args.output_dir, 'results')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, f'{model_name}.joblib')
    joblib.dump(model, model_path)
    logger.info(f'Model saved to {model_path}')
    
    # Save metrics and parameters
    model_info = {
        'name': model_name,
        'parameters': params,
        'cv_score': float(cv_score),
        'metrics': metrics,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metrics_path = os.path.join(results_dir, f'{model_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info(f'Metrics saved to {metrics_path}')

def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix for model evaluation.
    
    Args:
        cm: Confusion matrix
        model_name: Name of the model
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix: {model_name}')
    
    # Create directory for visualizations if it doesn't exist
    viz_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()

def generate_comparative_visualizations(all_metrics):
    """Generate visualizations comparing all model metrics.
    
    Args:
        all_metrics: Dictionary of metrics for all models
    """
    # Visualizations directory
    viz_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create DataFrame for easier plotting
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    model_names = list(all_metrics.keys())
    
    df = pd.DataFrame(columns=['model', 'metric', 'value'])
    
    for model in model_names:
        for metric in metrics:
            df = pd.concat([df, pd.DataFrame({
                'model': [model],
                'metric': [metric],
                'value': [all_metrics[model][metric]]
            })], ignore_index=True)
    
    # Bar chart comparing all metrics for all models
    plt.figure(figsize=(15, 10))
    sns.barplot(x='model', y='value', hue='metric', data=df)
    plt.title('Model Performance Comparison - All Metrics', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.ylim(0.85, 1.01)  # Adjust as needed based on your results
    plt.xticks(rotation=0)
    plt.legend(title='Metric', title_fontsize=12, fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'all_metrics_comparison.png'), dpi=300)
    plt.close()
    
    # Individual metrics comparison across models
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        metric_df = df[df['metric'] == metric]
        ax = sns.barplot(x='model', y='value', data=metric_df, hue='model', legend=False)
        
        # Add value labels on top of bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.4f}', 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha = 'center', va = 'bottom', fontsize=12)
        
        plt.title(f'{metric.replace("_", " ").title()} Comparison Across Models', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel(f'{metric.replace("_", " ").title()} Score', fontsize=14)
        
        # Set y-axis limit with some padding
        max_val = metric_df['value'].max()
        y_min = max(0.8, metric_df['value'].min() - 0.05)
        plt.ylim(y_min, min(1.01, max_val + 0.05))
        
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'{metric}_comparison.png'), dpi=300)
        plt.close()
    
    # Save summary table of metrics
    with open(os.path.join(viz_dir, 'metrics_summary.md'), 'w') as f:
        f.write('# Model Performance Metrics Summary\n\n')
        f.write('| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |\n')
        f.write('|-------|----------|-----------|--------|----------|--------|\n')
        
        for model in model_names:
            model_metrics = all_metrics[model]
            f.write(f"| {model.replace('_', ' ').title()} | {model_metrics['accuracy']:.4f} | {model_metrics['precision']:.4f} | {model_metrics['recall']:.4f} | {model_metrics['f1_score']:.4f} | {model_metrics['roc_auc']:.4f} |\n")
    
    logger.info(f'Comparative visualizations saved to {viz_dir}')

def main():
    """Main function to train and evaluate all baseline models."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up file logging
    log_file = os.path.join(args.output_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f'Starting baseline model training with seed {args.seed}')
    logger.info(f'Results will be saved to {args.output_dir}')
    
    # Load features
    X_train, X_test, y_train, y_test = load_features()
    
    # Train and evaluate all models
    all_metrics = {}
    
    # Logistic Regression
    lr_model, lr_params, lr_score = train_logistic_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, 'logistic_regression')
    save_model_and_metrics(lr_model, lr_metrics, lr_params, lr_score, 'logistic_regression')
    plot_confusion_matrix(np.array(lr_metrics['confusion_matrix']), 'LogisticRegression')
    all_metrics['logistic_regression'] = lr_metrics
    
    # SVM
    svm_model, svm_params, svm_score = train_svm(X_train, y_train)
    svm_metrics = evaluate_model(svm_model, X_test, y_test, 'svm')
    save_model_and_metrics(svm_model, svm_metrics, svm_params, svm_score, 'svm')
    plot_confusion_matrix(np.array(svm_metrics['confusion_matrix']), 'SVM')
    all_metrics['svm'] = svm_metrics
    
    # Random Forest
    rf_model, rf_params, rf_score = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, 'random_forest')
    save_model_and_metrics(rf_model, rf_metrics, rf_params, rf_score, 'random_forest')
    plot_confusion_matrix(np.array(rf_metrics['confusion_matrix']), 'RandomForest')
    all_metrics['random_forest'] = rf_metrics
    
    # XGBoost
    xgb_model, xgb_params, xgb_score = train_xgboost(X_train, y_train)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, 'xgboost')
    save_model_and_metrics(xgb_model, xgb_metrics, xgb_params, xgb_score, 'xgboost')
    plot_confusion_matrix(np.array(xgb_metrics['confusion_matrix']), 'XGBoost')
    all_metrics['xgboost'] = xgb_metrics
    
    # Generate comparative visualizations
    generate_comparative_visualizations(all_metrics)
    
    # Save overall results
    overall_results = {
        'models': {
            'logistic_regression': {
                'parameters': lr_params,
                'cv_score': float(lr_score),
                'metrics': lr_metrics
            },
            'svm': {
                'parameters': svm_params,
                'cv_score': float(svm_score),
                'metrics': svm_metrics
            },
            'random_forest': {
                'parameters': rf_params,
                'cv_score': float(rf_score),
                'metrics': rf_metrics
            },
            'xgboost': {
                'parameters': xgb_params,
                'cv_score': float(xgb_score),
                'metrics': xgb_metrics
            }
        },
        'best_model': max(all_metrics.items(), key=lambda x: x[1]['f1_score'])[0],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'seed': args.seed
    }
    
    overall_results_path = os.path.join(args.output_dir, 'overall_results.json')
    with open(overall_results_path, 'w') as f:
        json.dump(overall_results, f, indent=2)
    
    logger.info(f'Overall results saved to {overall_results_path}')
    
    # Print best model
    best_model = overall_results['best_model']
    best_f1 = all_metrics[best_model]['f1_score']
    logger.info('\n' + '=' * 50)
    logger.info(f'Best model: {best_model} with F1 score: {best_f1:.4f}')
    logger.info('=' * 50)

if __name__ == '__main__':
    args = parse_args()
    main() 