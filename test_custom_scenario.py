#!/usr/bin/env python
# -*- coding: utf-8 -*-


# This script tests a custom sleep scenario with both the regular and robust XGBoost models.
# It includes additional lifestyle factors like stress, exercise, caffeine, and alcohol.


import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from predict_waking_mood import create_derived_features, predict_mood

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Custom scenario with stress and exercise metrics
custom_metrics = {
    'total_sleep_time': 360,  # 6 hours
    'wake_time': 45,          # 45 min awake
    'rem_time': 72,           # 72 min (20% of sleep)
    'light_sleep_time': 198,  # 55% of sleep
    'deep_sleep_time': 90,    # 25% of sleep
    'sleep_efficiency': 89,   # 89% efficiency
    'rem_percentage': 20,     # 20% REM
    'rem_cycles': 3,          # 3 cycles
    'rem_awakenings': 3,      # 3 awakenings
    'stress_level': 7,        # High stress (scale 0-10)
    'exercise_minutes': 15,   # Low exercise 
    'caffeine_mg': 200,       # High caffeine
    'alcohol_units': 2        # Moderate alcohol intake
}

logger.info('Testing custom scenario with stress and other lifestyle factors')
logger.info('Sleep Metrics:')
for key, value in custom_metrics.items():
    logger.info(f'  {key}: {value}')

# Test standard XGBoost
standard_result = predict_mood(custom_metrics, 'xgboost')
logger.info('\nStandard XGBoost prediction:')
if standard_result:
    logger.info(f'  Predicted Mood: {"Good" if standard_result["good_mood"] else "Bad"}')
    logger.info(f'  Confidence: {standard_result["good_mood_probability"]:.2f}')
    logger.info(f'  Mood Score (0-10): {standard_result["mood_score"]:.1f}')

# Test robust XGBoost  
robust_result = predict_mood(custom_metrics, 'xgboost_robust')
logger.info('\nRobust XGBoost prediction:')
if robust_result:
    logger.info(f'  Predicted Mood: {"Good" if robust_result["good_mood"] else "Bad"}')
    logger.info(f'  Confidence: {robust_result["good_mood_probability"]:.2f}')
    logger.info(f'  Mood Score (0-10): {robust_result["mood_score"]:.1f}')

# Also try testing with different stress levels
low_stress_metrics = custom_metrics.copy()
low_stress_metrics['stress_level'] = 2
low_stress_metrics['exercise_minutes'] = 60

logger.info('\n\nTesting with low stress and high exercise:')
logger.info('  stress_level: 2')
logger.info('  exercise_minutes: 60')

# Test robust XGBoost with low stress  
robust_result_low_stress = predict_mood(low_stress_metrics, 'xgboost_robust')
logger.info('\nRobust XGBoost prediction (low stress):')
if robust_result_low_stress:
    logger.info(f'  Predicted Mood: {"Good" if robust_result_low_stress["good_mood"] else "Bad"}')
    logger.info(f'  Confidence: {robust_result_low_stress["good_mood_probability"]:.2f}')
    logger.info(f'  Mood Score (0-10): {robust_result_low_stress["mood_score"]:.1f}') 