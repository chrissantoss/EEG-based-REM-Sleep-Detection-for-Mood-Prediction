#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script visualizes hyperparameter tuning results to compare performance before and after tuning.
It generates plots and tables to help understand the impact of hyperparameter tuning.
"""

import os
import sys
import logging
import argparse
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Define the data directory
ROOT_DIR = Path(__file__).resolve().parents[2]
TUNING_DIR = ROOT_DIR / "models" / "tuning_results"
VISUALIZATION_DIR = ROOT_DIR / "visualizations" / "tuning_results"

# Ensure directories exist
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

def load_tuning_results(task=None, model_name=None):
    """
    Load hyperparameter tuning results from JSON files.
    
    Args:
        task (str, optional): Filter by task name
        model_name (str, optional): Filter by model name
    
    Returns:
        list: List of tuning result dictionaries
    """
    # Determine search path
    if task:
        search_path = TUNING_DIR / task / "*.json"
    else:
        search_path = TUNING_DIR / "**" / "*.json"
    
    # Find all JSON files
    result_files = glob.glob(str(search_path), recursive=True)
    
    if not result_files:
        logger.error(f"No tuning result files found at {search_path}")
        return []
    
    # Load results
    results = []
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
                
                # Filter by model name if specified
                if model_name and result.get("model_name") != model_name:
                    continue
                
                # Add file path to result
                result["file_path"] = file_path
                
                # Extract task from file path if not in result
                if "task" not in result:
                    task_dir = Path(file_path).parent.name
                    result["task"] = task_dir
                
                results.append(result)
        
        except Exception as e:
            logger.error(f"Error loading tuning results from {file_path}: {e}")
    
    logger.info(f"Loaded {len(results)} tuning result files")
    return results

def create_comparison_dataframe(results):
    """
    Create a DataFrame comparing performance before and after tuning.
    
    Args:
        results (list): List of tuning result dictionaries
    
    Returns:
        pd.DataFrame: DataFrame with comparison metrics
    """
    if not results:
        return pd.DataFrame()
    
    # Extract relevant data
    data = []
    for result in results:
        model_name = result.get("model_name", "unknown")
        task = result.get("task", "unknown")
        timestamp = result.get("timestamp", "unknown")
        
        # Get metrics
        before_metrics = result.get("before_metrics", {})
        after_metrics = result.get("after_metrics", {})
        improvement = result.get("improvement", {})
        
        # Create row
        row = {
            "model_name": model_name,
            "task": task,
            "timestamp": timestamp,
            "before_accuracy": before_metrics.get("accuracy", 0),
            "after_accuracy": after_metrics.get("accuracy", 0),
            "before_precision": before_metrics.get("precision", 0),
            "after_precision": after_metrics.get("precision", 0),
            "before_recall": before_metrics.get("recall", 0),
            "after_recall": after_metrics.get("recall", 0),
            "before_f1": before_metrics.get("f1", 0),
            "after_f1": after_metrics.get("f1", 0),
            "before_roc_auc": before_metrics.get("roc_auc", 0),
            "after_roc_auc": after_metrics.get("roc_auc", 0),
            "f1_improvement": improvement.get("f1_absolute", 0),
            "f1_improvement_percent": improvement.get("f1_percent", 0),
            "tuning_time": result.get("tuning_time", 0),
            "n_iter": result.get("n_iter", 0),
            "cv": result.get("cv", 0),
            "scoring": result.get("scoring", "unknown"),
            "file_path": result.get("file_path", "unknown")
        }
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Convert timestamp to datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    return df

def plot_metric_comparison(df, metric="f1", output_dir=None):
    """
    Plot before and after comparison for a specific metric.
    
    Args:
        df (pd.DataFrame): DataFrame with comparison metrics
        metric (str): Metric to plot (e.g., "f1", "accuracy")
        output_dir (Path, optional): Directory to save the plot
    
    Returns:
        Path: Path to the saved plot
    """
    if df.empty:
        logger.error("Cannot create plot: DataFrame is empty")
        return None
    
    # Check if metric columns exist
    before_col = f"before_{metric}"
    after_col = f"after_{metric}"
    
    if before_col not in df.columns or after_col not in df.columns:
        logger.error(f"Metric columns not found: {before_col}, {after_col}")
        return None
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create grouped bar chart
    models = df["model_name"].unique()
    tasks = df["task"].unique()
    
    # Set up plot
    x = np.arange(len(models))
    width = 0.35
    
    # Plot for each task
    for i, task in enumerate(tasks):
        task_df = df[df["task"] == task]
        
        # Group by model and get the latest result for each model
        latest_results = task_df.sort_values("timestamp").groupby("model_name").last()
        
        # Ensure all models are included
        model_data = []
        for model in models:
            if model in latest_results.index:
                model_data.append(latest_results.loc[model])
            else:
                # Create empty row
                empty_row = pd.Series({before_col: 0, after_col: 0, "model_name": model})
                model_data.append(empty_row)
        
        task_df = pd.DataFrame(model_data)
        
        # Plot bars
        offset = width * (i - len(tasks) / 2 + 0.5)
        plt.bar(x + offset, task_df[before_col], width, label=f"{task} (Before)")
        plt.bar(x + offset + width/2, task_df[after_col], width, label=f"{task} (After)")
    
    # Add labels and legend
    plt.xlabel("Model")
    plt.ylabel(f"{metric.upper()} Score")
    plt.title(f"{metric.upper()} Score Before and After Hyperparameter Tuning")
    plt.xticks(x, models)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Save plot
    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = output_dir / f"{metric}_comparison_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved plot to {plot_file}")
        return plot_file
    else:
        plt.show()
        plt.close()
        return None

def plot_improvement_comparison(df, output_dir=None):
    """
    Plot improvement percentage for each model and task.
    
    Args:
        df (pd.DataFrame): DataFrame with comparison metrics
        output_dir (Path, optional): Directory to save the plot
    
    Returns:
        Path: Path to the saved plot
    """
    if df.empty:
        logger.error("Cannot create plot: DataFrame is empty")
        return None
    
    # Check if improvement column exists
    if "f1_improvement_percent" not in df.columns:
        logger.error("Improvement column not found: f1_improvement_percent")
        return None
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create heatmap data
    pivot_df = df.pivot_table(
        index="model_name", 
        columns="task", 
        values="f1_improvement_percent",
        aggfunc="max"  # Use max improvement for each model-task combination
    )
    
    # Create heatmap
    sns.heatmap(
        pivot_df, 
        annot=True, 
        fmt=".2f", 
        cmap="RdYlGn", 
        linewidths=0.5,
        cbar_kws={"label": "F1 Score Improvement (%)"}
    )
    
    # Add labels
    plt.title("F1 Score Improvement (%) After Hyperparameter Tuning")
    plt.tight_layout()
    
    # Save plot
    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = output_dir / f"improvement_heatmap_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved plot to {plot_file}")
        return plot_file
    else:
        plt.show()
        plt.close()
        return None

def plot_parameter_importance(results, model_name, task, output_dir=None):
    """
    Plot parameter importance for a specific model and task.
    
    Args:
        results (list): List of tuning result dictionaries
        model_name (str): Model name to plot
        task (str): Task name to plot
        output_dir (Path, optional): Directory to save the plot
    
    Returns:
        Path: Path to the saved plot
    """
    # Filter results
    filtered_results = [r for r in results if r.get("model_name") == model_name and r.get("task") == task]
    
    if not filtered_results:
        logger.error(f"No results found for model {model_name} and task {task}")
        return None
    
    # Get the latest result
    latest_result = max(filtered_results, key=lambda r: r.get("timestamp", ""))
    
    # Get best parameters
    best_params = latest_result.get("best_params", {})
    
    if not best_params:
        logger.error("No best parameters found in the result")
        return None
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot parameters
    param_names = list(best_params.keys())
    param_values = [str(best_params[p]) for p in param_names]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(param_names))
    plt.barh(y_pos, np.ones(len(param_names)), align="center", alpha=0.5)
    plt.yticks(y_pos, param_names)
    
    # Add parameter values as text
    for i, value in enumerate(param_values):
        plt.text(0.5, i, str(value), ha="center", va="center")
    
    # Add labels
    plt.xlabel("Best Value")
    plt.title(f"Best Parameters for {model_name} on {task}")
    plt.tight_layout()
    
    # Save plot
    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = output_dir / f"best_params_{model_name}_{task}_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved plot to {plot_file}")
        return plot_file
    else:
        plt.show()
        plt.close()
        return None

def generate_report(df, results, output_dir):
    """
    Generate a comprehensive HTML report of tuning results.
    
    Args:
        df (pd.DataFrame): DataFrame with comparison metrics
        results (list): List of tuning result dictionaries
        output_dir (Path): Directory to save the report
    
    Returns:
        Path: Path to the saved report
    """
    if df.empty:
        logger.error("Cannot create report: DataFrame is empty")
        return None
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"tuning_report_{timestamp}.html"
    
    # Start HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hyperparameter Tuning Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .improvement-positive {{ color: green; }}
            .improvement-negative {{ color: red; }}
            .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>Hyperparameter Tuning Report</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>Total tuning runs: {len(results)}</p>
            <p>Models tuned: {', '.join(df['model_name'].unique())}</p>
            <p>Tasks: {', '.join(df['task'].unique())}</p>
        </div>
        
        <h2>Performance Comparison</h2>
    """
    
    # Add performance comparison table
    comparison_df = df.sort_values(["task", "model_name", "timestamp"]).groupby(["task", "model_name"]).last()
    comparison_df = comparison_df.reset_index()
    
    # Select relevant columns
    table_columns = [
        "task", "model_name", 
        "before_f1", "after_f1", "f1_improvement_percent",
        "before_accuracy", "after_accuracy",
        "before_precision", "after_precision",
        "before_recall", "after_recall",
        "tuning_time"
    ]
    
    table_df = comparison_df[table_columns].copy()
    
    # Format numeric columns
    for col in table_df.columns:
        if col in ["task", "model_name"]:
            continue
        if "percent" in col:
            table_df[col] = table_df[col].map(lambda x: f"{x:.2f}%")
        elif "time" in col:
            table_df[col] = table_df[col].map(lambda x: f"{x:.1f}s")
        else:
            table_df[col] = table_df[col].map(lambda x: f"{x:.4f}")
    
    # Rename columns for display
    table_df.columns = [
        "Task", "Model", 
        "F1 (Before)", "F1 (After)", "F1 Improvement",
        "Accuracy (Before)", "Accuracy (After)",
        "Precision (Before)", "Precision (After)",
        "Recall (Before)", "Recall (After)",
        "Tuning Time"
    ]
    
    # Add table to HTML
    html_content += table_df.to_html(index=False, classes="table table-striped")
    
    # Add best parameters section
    html_content += """
        <h2>Best Parameters</h2>
    """
    
    # Group results by model and task
    for model_name in df["model_name"].unique():
        for task in df["task"].unique():
            # Filter results
            filtered_results = [r for r in results if r.get("model_name") == model_name and r.get("task") == task]
            
            if not filtered_results:
                continue
            
            # Get the latest result
            latest_result = max(filtered_results, key=lambda r: r.get("timestamp", ""))
            
            # Get best parameters
            best_params = latest_result.get("best_params", {})
            
            if not best_params:
                continue
            
            # Add section
            html_content += f"""
                <h3>{model_name} - {task}</h3>
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
            """
            
            # Add parameters
            for param, value in best_params.items():
                html_content += f"""
                    <tr>
                        <td>{param}</td>
                        <td>{value}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            """
    
    # Close HTML
    html_content += """
    </body>
    </html>
    """
    
    # Write to file
    with open(report_file, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Saved report to {report_file}")
    return report_file

def main():
    """Main function to visualize tuning results."""
    parser = argparse.ArgumentParser(description="Visualize hyperparameter tuning results")
    parser.add_argument(
        "--task", 
        help="Filter by task name"
    )
    parser.add_argument(
        "--model", 
        help="Filter by model name"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save visualizations (default: visualizations/tuning_results)"
    )
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = VISUALIZATION_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tuning results
    results = load_tuning_results(args.task, args.model)
    
    if not results:
        logger.error("No tuning results found")
        return 1
    
    # Create comparison DataFrame
    df = create_comparison_dataframe(results)
    
    if df.empty:
        logger.error("Failed to create comparison DataFrame")
        return 1
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    # Plot metric comparisons
    for metric in ["f1", "accuracy", "precision", "recall", "roc_auc"]:
        plot_metric_comparison(df, metric, output_dir)
    
    # Plot improvement comparison
    plot_improvement_comparison(df, output_dir)
    
    # Plot parameter importance for each model and task
    for model_name in df["model_name"].unique():
        for task in df["task"].unique():
            plot_parameter_importance(results, model_name, task, output_dir)
    
    # Generate report
    report_file = generate_report(df, results, output_dir)
    
    if report_file:
        logger.info(f"Report generated: {report_file}")
    
    logger.info("Visualization completed")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 