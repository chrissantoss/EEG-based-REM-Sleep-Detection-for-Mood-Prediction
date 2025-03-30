#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script opens the latest hyperparameter tuning visualization report in the default browser.
"""

import os
import sys
import glob
import webbrowser
from pathlib import Path

def open_latest_report():
    """Open the latest tuning report in the default browser."""
    # Find the latest report
    report_dir = Path(__file__).resolve().parent / "visualizations" / "tuning_results"
    reports = glob.glob(str(report_dir / "tuning_report_*.html"))
    
    if not reports:
        print("No visualization reports found!")
        return 1
    
    # Get the latest report
    latest_report = max(reports, key=os.path.getctime)
    print(f"Opening report: {latest_report}")
    
    # Open in browser
    webbrowser.open(f"file://{latest_report}", new=2)
    
    # Display additional information
    print("\nVisualization report opened in your default browser.")
    print("The report shows the following:")
    print("1. Performance metrics before and after hyperparameter tuning")
    print("2. Improvement percentages for each model and task")
    print("3. Best hyperparameter values found during tuning")
    print("\nTo view all visualization files, check the directory:")
    print(f"{report_dir}\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(open_latest_report()) 