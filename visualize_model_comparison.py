#!/usr/bin/env python
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set up the styles for the plots
plt.style.use('ggplot')
sns.set_palette("colorblind")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def visualize_model_comparison(results_dict=None):
    """
    Create visualizations to compare model performance metrics.
    
    Args:
        results_dict: Optional dictionary containing model results. If None, uses the data
                     from the most recent test run.
    """
    if results_dict is None:
        # Sample results based on the test output
        results_dict = {
            'logistic_regression': {
                'accuracy': 0.9121,
                'precision': 0.9538,
                'recall': 0.9254,
                'f1_score': 0.9394,
                'roc_auc': 0.9782
            },
            'svm': {
                'accuracy': 0.9341,
                'precision': 0.9420,
                'recall': 0.9701,
                'f1_score': 0.9559,
                'roc_auc': 0.9764
            },
            'random_forest': {
                'accuracy': 0.9121,
                'precision': 0.8933,
                'recall': 1.0000,
                'f1_score': 0.9437,
                'roc_auc': 0.9583
            },
            'xgboost': {
                'accuracy': 0.9341,
                'precision': 0.9178,
                'recall': 1.0000,
                'f1_score': 0.9571,
                'roc_auc': 0.9888
            }
        }
    
    # Convert to DataFrame for easier plotting
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    model_names = list(results_dict.keys())
    
    df = pd.DataFrame(columns=['model', 'metric', 'value'])
    
    for model in model_names:
        for metric in metrics:
            df = pd.concat([df, pd.DataFrame({
                'model': [model],
                'metric': [metric],
                'value': [results_dict[model][metric]]
            })], ignore_index=True)
    
    # Create directory for visualizations if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Bar chart comparing all metrics for all models
    plt.figure(figsize=(15, 10))
    sns.barplot(x='model', y='value', hue='metric', data=df)
    plt.title('Model Performance Comparison - All Metrics', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.ylim(0.85, 1.01)  # Adjust as needed based on your results
    plt.xticks(rotation=0)
    plt.legend(title='Metric', title_fontsize=12, fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.savefig('visualizations/all_metrics_comparison.png', dpi=300, bbox_inches='tight')
    
    # 2. Individual metrics comparison across models
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        metric_df = df[df['metric'] == metric]
        ax = sns.barplot(x='model', y='value', data=metric_df, palette='viridis')
        
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
        plt.savefig(f'visualizations/{metric}_comparison.png', dpi=300, bbox_inches='tight')
    
    # 3. Radar chart for comparing all models
    # Create data for radar chart
    labels = metrics
    num_models = len(model_names)
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for i, model in enumerate(model_names):
        values = [results_dict[model][metric] for metric in metrics]
        values += values[:1]  # Close the polygon
        
        # Plot the values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Set labels and title
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", " ").title() for m in labels])
    ax.set_yticks([0.85, 0.9, 0.95, 1.0])
    ax.set_yticklabels(["0.85", "0.9", "0.95", "1.0"])
    ax.set_ylim(0.85, 1.01)
    
    plt.title('Model Performance Metrics Comparison', fontsize=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig('visualizations/radar_chart_comparison.png', dpi=300, bbox_inches='tight')
    
    # 4. Create a summary table image
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Reshape data for the table
    table_data = []
    for model in model_names:
        row = [model.replace('_', ' ').title()]
        for metric in metrics:
            row.append(f"{results_dict[model][metric]:.4f}")
        table_data.append(row)
    
    column_labels = ['Model'] + [m.replace('_', ' ').title() for m in metrics]
    table = ax.table(cellText=table_data, colLabels=column_labels, loc='center', cellLoc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Color the header row
    for i, key in enumerate(column_labels):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#4472C4')
    
    # Highlight the best value for each metric
    best_indices = {}
    for i, metric in enumerate(metrics):
        values = [results_dict[model][metric] for model in model_names]
        if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        best_indices[(best_idx + 1, i + 1)] = True  # +1 because of header row and first column
    
    # Apply highlight to best values
    for (row_idx, col_idx) in best_indices:
        cell = table[(row_idx, col_idx)]
        cell.set_facecolor('#E2EFDA')  # Light green
    
    plt.title('Model Performance Metrics Summary', fontsize=16)
    plt.tight_layout()
    plt.savefig('visualizations/metrics_summary_table.png', dpi=300, bbox_inches='tight')
    
    print(f"Visualizations saved to 'visualizations/' directory.")
    
    return df  # Return the DataFrame for further analysis if needed

if __name__ == "__main__":
    visualize_model_comparison() 