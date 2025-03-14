#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script tests the functionality of the entire EEG-based REM sleep detection pipeline.
It verifies that each component works correctly and produces the expected outputs.
"""

import os
import sys
import unittest
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Import project modules
from data.download_datasets import download_dataset, generate_synthetic_dataset
from data.process_data import filter_eeg_data, segment_eeg_data, extract_features, normalize_features
from features.extract_features import select_features, prepare_rem_detection_features, prepare_mood_prediction_features
from models.train_model import train_model, evaluate_model

class TestDataDownload(unittest.TestCase):
    """Test the data download functionality."""
    
    def setUp(self):
        """Set up the test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up the test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_generate_synthetic_dataset(self):
        """Test the synthetic dataset generation."""
        # Generate a synthetic dataset
        dataset_file = generate_synthetic_dataset(self.temp_dir)
        
        # Check if the file was created
        self.assertTrue(dataset_file.exists())
        
        # Check if the file is a valid CSV
        df = pd.read_csv(dataset_file)
        
        # Check if the DataFrame has the expected columns
        expected_columns = [
            'subject_id', 'night', 'date', 'total_sleep_time', 'wake_time',
            'rem_time', 'light_sleep_time', 'deep_sleep_time', 'sleep_efficiency',
            'rem_percentage', 'rem_cycles', 'rem_awakenings', 'mood_rating', 'anxiety_rating'
        ]
        
        for col in expected_columns:
            self.assertIn(col, df.columns)
        
        # Check if the DataFrame has the expected number of rows
        self.assertEqual(len(df), 20 * 5)  # 20 subjects, 5 nights each

class TestDataProcessing(unittest.TestCase):
    """Test the data processing functionality."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a simple synthetic EEG data
        self.sfreq = 100  # 100 Hz
        self.duration = 10  # 10 seconds
        self.n_channels = 2
        
        # Generate random EEG data
        np.random.seed(42)
        self.eeg_data = np.random.randn(self.n_channels, self.sfreq * self.duration)
        
        # Create a mock MNE Raw object
        class MockRaw:
            def __init__(self, data, sfreq):
                self.data = data
                self.info = {'sfreq': sfreq}
            
            def get_data(self, return_times=False):
                if return_times:
                    times = np.arange(self.data.shape[1]) / self.info['sfreq']
                    return self.data, times
                return self.data
            
            def copy(self):
                return MockRaw(self.data.copy(), self.info['sfreq'])
            
            def filter(self, l_freq=None, h_freq=None):
                # Simple mock filter (just return self)
                return self
        
        self.raw = MockRaw(self.eeg_data, self.sfreq)
    
    def test_segment_eeg_data(self):
        """Test the EEG data segmentation."""
        # Segment the data
        window_size = 2.0  # 2 seconds
        segments = segment_eeg_data(self.raw, window_size=window_size, overlap=0.0)
        
        # Check if the segments have the expected shape
        expected_segments = int(self.duration / window_size)
        self.assertEqual(len(segments), expected_segments)
        
        # Check if each segment has the expected shape
        window_samples = int(window_size * self.sfreq)
        for segment in segments:
            self.assertEqual(segment.shape, (self.n_channels, window_samples))
    
    def test_extract_features(self):
        """Test the feature extraction."""
        # Segment the data
        window_size = 2.0  # 2 seconds
        segments = segment_eeg_data(self.raw, window_size=window_size, overlap=0.0)
        
        # Extract features
        features_df = extract_features(segments, self.sfreq)
        
        # Check if the DataFrame has the expected number of rows
        expected_segments = int(self.duration / window_size)
        self.assertEqual(len(features_df), expected_segments)
        
        # Check if the DataFrame has the expected columns
        expected_columns = [
            'mean_ch0', 'std_ch0', 'min_ch0', 'max_ch0', 'ptp_ch0',
            'skew_ch0', 'kurtosis_ch0', 'delta_power_ch0', 'theta_power_ch0',
            'alpha_power_ch0', 'beta_power_ch0', 'segment_idx'
        ]
        
        for col in expected_columns:
            self.assertIn(col, features_df.columns)
    
    def test_normalize_features(self):
        """Test the feature normalization."""
        # Create a simple DataFrame
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'segment_idx': [0, 1, 2, 3, 4]
        })
        
        # Normalize the features
        normalized_df = normalize_features(df)
        
        # Check if the DataFrame has the same shape
        self.assertEqual(normalized_df.shape, df.shape)
        
        # Check if the numerical columns have zero mean and unit variance
        for col in ['feature1', 'feature2']:
            self.assertAlmostEqual(normalized_df[col].mean(), 0.0, places=10)
            self.assertAlmostEqual(normalized_df[col].std(), 1.0, places=10)
        
        # Check if the segment_idx column is unchanged
        pd.testing.assert_series_equal(normalized_df['segment_idx'], df['segment_idx'])

class TestFeatureExtraction(unittest.TestCase):
    """Test the feature extraction functionality."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a simple synthetic dataset
        np.random.seed(42)
        n_samples = 100
        
        # Create features
        self.X = pd.DataFrame({
            f'feature{i}': np.random.randn(n_samples) for i in range(10)
        })
        
        # Create binary target
        self.y = pd.Series(np.random.randint(0, 2, n_samples))
    
    def test_select_features(self):
        """Test the feature selection."""
        # Select features using ANOVA
        X_selected, selected_features = select_features(self.X, self.y, method='anova', k=5)
        
        # Check if the selected features have the expected shape
        self.assertEqual(X_selected.shape, (len(self.X), 5))
        self.assertEqual(len(selected_features), 5)
        
        # Check if the selected features are a subset of the original features
        for feature in selected_features:
            self.assertIn(feature, self.X.columns)
    
    def test_prepare_rem_detection_features(self):
        """Test the REM detection feature preparation."""
        # Create a synthetic dataset
        data = self.X.copy()
        data['sleep_stage'] = np.random.randint(0, 5, len(data))
        
        # Prepare features
        features_dict = prepare_rem_detection_features(data)
        
        # Check if the dictionary has the expected keys
        expected_keys = ['X_train', 'X_test', 'y_train', 'y_test', 'selected_features', 'scaler', 'feature_cols']
        for key in expected_keys:
            self.assertIn(key, features_dict)
        
        # Check if the train/test split has the expected shape
        self.assertEqual(len(features_dict['X_train']) + len(features_dict['X_test']), len(data))
        self.assertEqual(len(features_dict['y_train']) + len(features_dict['y_test']), len(data))

class TestModelTraining(unittest.TestCase):
    """Test the model training functionality."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a simple synthetic dataset
        np.random.seed(42)
        n_samples = 100
        
        # Create features
        self.X = pd.DataFrame({
            f'feature{i}': np.random.randn(n_samples) for i in range(5)
        })
        
        # Create binary target
        self.y = pd.Series(np.random.randint(0, 2, n_samples))
    
    def test_train_logistic_regression(self):
        """Test training a logistic regression model."""
        # Define a simple parameter grid
        params = {
            'C': [0.1, 1.0],
            'penalty': ['l2'],
            'solver': ['liblinear'],
            'class_weight': [None]
        }
        
        # Train the model
        model, best_params = train_model('logistic_regression', self.X, self.y, params=params, cv=2)
        
        # Check if the model and best parameters are returned
        self.assertIsNotNone(model)
        self.assertIsNotNone(best_params)
        
        # Check if the best parameters are a subset of the parameter grid
        for key, value in best_params.items():
            self.assertIn(key, params)
            self.assertIn(value, params[key])
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        # Train a simple model
        params = {
            'C': [1.0],
            'penalty': ['l2'],
            'solver': ['liblinear'],
            'class_weight': [None]
        }
        
        model, _ = train_model('logistic_regression', self.X, self.y, params=params, cv=2)
        
        # Evaluate the model
        metrics = evaluate_model(model, self.X, self.y)
        
        # Check if the metrics dictionary has the expected keys
        expected_keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'confusion_matrix']
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Check if the metrics have valid values
        for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)

if __name__ == '__main__':
    unittest.main() 