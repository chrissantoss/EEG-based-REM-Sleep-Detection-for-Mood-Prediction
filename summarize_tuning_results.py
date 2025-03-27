#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script summarizes the hyperparameter tuning results for all models and tasks.
"""

import os
import json
from pathlib import Path
import pandas as pd

# Define the root directory
ROOT_DIR = Path(__file__).resolve().parent
TUNING_DIR = ROOT_DIR / "models" / "tuning_results"

def load_tuning_results():
    """
    Load all tuning results from the tuning_results directory.
    
    Returns:
        list: List of tuning result dictionaries
    """
    results = []
    
    # Walk through the tuning_results directory
    for task_dir in TUNING_DIR.iterdir():
        if not task_dir.is_dir() or task_dir.name.startswith('.'):
            continue
        
        task_name = task_dir.name
        
        # Look for tuning result files
        for result_file in task_dir.glob("*_tuning_results_*.json"):
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                
                # Add the task name and file path
                result_data["task_name"] = task_name
                result_data["file_path"] = str(result_file)
                
                results.append(result_data)
            except Exception as e:
                print(f"Error loading {result_file}: {e}")
    
    return results

def summarize_results(results):
    """
    Summarize the tuning results.
    
    Args:
        results (list): List of tuning result dictionaries
    """
    if not results:
        print("No tuning results found.")
        return
    
    # Create a summary table
    summary_data = []
    
    for result in results:
        model_name = result.get("model_name", "Unknown")
        task_name = result.get("task_name", "Unknown")
        
        before_metrics = result.get("before_metrics", {})
        after_metrics = result.get("after_metrics", {})
        
        # Get metrics
        before_f1 = before_metrics.get("f1", 0)
        after_f1 = after_metrics.get("f1", 0)
        f1_improvement = after_f1 - before_f1
        f1_improvement_pct = (f1_improvement / before_f1 * 100) if before_f1 > 0 else 0
        
        before_accuracy = before_metrics.get("accuracy", 0)
        after_accuracy = after_metrics.get("accuracy", 0)
        accuracy_improvement = after_accuracy - before_accuracy
        accuracy_improvement_pct = (accuracy_improvement / before_accuracy * 100) if before_accuracy > 0 else 0
        
        # Get best parameters
        best_params = result.get("best_params", {})
        
        # Add to summary data
        summary_data.append({
            "Model": model_name,
            "Task": task_name,
            "Before F1": before_f1,
            "After F1": after_f1,
            "F1 Improvement": f1_improvement,
            "F1 Improvement %": f1_improvement_pct,
            "Before Accuracy": before_accuracy,
            "After Accuracy": after_accuracy,
            "Accuracy Improvement": accuracy_improvement,
            "Accuracy Improvement %": accuracy_improvement_pct,
            "Best Parameters": str(best_params)
        })
    
    # Create a DataFrame
    df = pd.DataFrame(summary_data)
    
    # Sort by F1 improvement
    df = df.sort_values(by=["Task", "F1 Improvement"], ascending=[True, False])
    
    # Print the summary table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 60)
    
    print("\n=== Hyperparameter Tuning Results Summary ===\n")
    print(df[["Model", "Task", "Before F1", "After F1", "F1 Improvement", "F1 Improvement %", 
              "Before Accuracy", "After Accuracy", "Accuracy Improvement", "Accuracy Improvement %"]].to_string(index=False))
    
    # Print the best model for each task
    print("\n=== Best Model for Each Task ===\n")
    
    best_models = {}
    for task in df["Task"].unique():
        task_df = df[df["Task"] == task]
        best_model_row = task_df.iloc[task_df["F1 Improvement"].idxmax()]
        
        best_models[task] = {
            "Model": best_model_row["Model"],
            "F1 Score": best_model_row["After F1"],
            "Accuracy": best_model_row["After Accuracy"],
            "Best Parameters": best_model_row["Best Parameters"]
        }
        
        print(f"Task: {task}")
        print(f"  Best Model: {best_model_row['Model']}")
        print(f"  F1 Score: {best_model_row['After F1']:.4f}")
        print(f"  Accuracy: {best_model_row['After Accuracy']:.4f}")
        print(f"  Parameters: {best_model_row['Best Parameters']}\n")

def main():
    """Main function to summarize tuning results."""
    results = load_tuning_results()
    summarize_results(results)

if __name__ == "__main__":
    main() 