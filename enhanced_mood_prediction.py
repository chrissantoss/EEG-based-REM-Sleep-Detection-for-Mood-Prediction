#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Enhances mood prediction by incorporating additional lifestyle factors beyond sleep metrics.
# Includes stress, exercise, diet, and caffeine consumption data for improved accuracy.

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Define directories
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
ENHANCED_DIR = MODELS_DIR / "enhanced_models"

# Create necessary directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ENHANCED_DIR.mkdir(parents=True, exist_ok=True)

# Define additional factors that may influence mood
ADDITIONAL_FACTORS = [
    'stress_level',           # 0-10 scale
    'exercise_minutes',       # Minutes of exercise
    'caffeine_mg',            # Milligrams of caffeine
    'screen_time_minutes',    # Minutes of screen time before bed
    'alcohol_units',          # Units of alcohol consumed
    'outdoor_time_minutes',   # Minutes spent outdoors
    'social_interaction_score', # 0-10 scale of social interaction
    'meditation_minutes'      # Minutes spent meditating
]

class EnhancedMoodPredictor:
    # Handles enhanced mood prediction with additional lifestyle factors
    
    def __init__(self, base_model_name="xgboost"):
        # Initializes the enhanced mood predictor with specified base model
        self.base_model_name = base_model_name
        
        # Load base model
        self.base_model, self.base_metadata = self.load_base_model()
        
        # Initialize enhanced model
        self.enhanced_model = None
        self.enhanced_metadata = None
        
        # Path for enhanced model
        self.enhanced_model_path = ENHANCED_DIR / f"{base_model_name}_enhanced.joblib"
        self.enhanced_metadata_path = ENHANCED_DIR / f"{base_model_name}_enhanced_metadata.joblib"
        
        # Try to load existing enhanced model
        if self.enhanced_model_path.exists() and self.enhanced_metadata_path.exists():
            try:
                self.enhanced_model = joblib.load(self.enhanced_model_path)
                self.enhanced_metadata = joblib.load(self.enhanced_metadata_path)
                logger.info(f"Loaded enhanced model: {self.enhanced_model_path}")
            except Exception as e:
                logger.error(f"Error loading enhanced model: {e}")
    
    def load_base_model(self):
        # Loads the base model for mood prediction
        model_path = MODELS_DIR / "mood_prediction" / f"{self.base_model_name}.joblib"
        metadata_path = MODELS_DIR / "mood_prediction" / f"{self.base_model_name}_metadata.joblib"
        
        # Check for refined models first
        refined_model_path = MODELS_DIR / "mood_prediction" / f"{self.base_model_name}_refined.joblib"
        refined_metadata_path = MODELS_DIR / "mood_prediction" / f"{self.base_model_name}_refined_metadata.joblib"
        
        if refined_model_path.exists() and refined_metadata_path.exists():
            model_path = refined_model_path
            metadata_path = refined_metadata_path
        
        if not model_path.exists() or not metadata_path.exists():
            logger.error(f"Base model not found: {model_path}")
            return None, None
        
        try:
            model = joblib.load(model_path)
            metadata = joblib.load(metadata_path)
            
            # Ensure metadata has a 'features' key
            if 'features' not in metadata and 'selected_features' in metadata:
                metadata['features'] = metadata['selected_features']
            elif 'features' not in metadata and hasattr(model, 'feature_names_in_'):
                metadata['features'] = list(model.feature_names_in_)
            elif 'features' not in metadata:
                # Try to guess features from model
                if hasattr(model, 'feature_importances_'):
                    # Create basic sleep metrics features
                    basic_features = [
                        'total_sleep_time', 'wake_time', 'rem_time', 
                        'light_sleep_time', 'deep_sleep_time', 'sleep_efficiency', 
                        'rem_percentage'
                    ]
                    metadata['features'] = basic_features
                    logger.warning(f"Guessed features for base model: {basic_features}")
            
            logger.info(f"Loaded base model: {model_path}")
            return model, metadata
        
        except Exception as e:
            logger.error(f"Error loading base model: {e}")
            return None, None
    
    def train_enhanced_model(self, X_train, y_train, X_test, y_test):
        # Trains an enhanced model using sleep features and additional lifestyle factors
        logger.info("Training enhanced mood prediction model")
        
        # Store continuous mood score for analysis if available
        mood_score_train = None
        mood_score_test = None
        if 'mood_score' in X_train.columns:
            mood_score_train = X_train['mood_score'].copy()
            mood_score_test = X_test['mood_score'].copy()
            
            # Remove mood_score from features
            X_train = X_train.drop('mood_score', axis=1)
            X_test = X_test.drop('mood_score', axis=1)
        
        # Determine which model to use based on base model
        if self.base_model_name == "xgboost" or self.base_model_name.startswith("xgboost_"):
            # Use XGBoost
            model = xgb.XGBClassifier(
                learning_rate=0.05,
                max_depth=4,
                n_estimators=200,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            # Use Random Forest
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        
        # Store feature names before training
        feature_names = list(X_train.columns)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_prob))
        }
        
        logger.info("Performance metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Calculate feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1]
            features = X_train.columns
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': features,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            logger.info("\nTop 10 most important features:")
            for i, feature in enumerate(importance_df['feature'][:10]):
                logger.info(f"  {i+1}. {feature}: {importance_df['importance'].iloc[i]:.4f}")
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            plt.barh(features[indices][:15], importance[indices][:15])
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('Feature Importance for Enhanced Mood Prediction')
            plt.tight_layout()
            
            # Save plot
            importance_plot_path = ENHANCED_DIR / f"{self.base_model_name}_importance.png"
            plt.savefig(importance_plot_path)
            logger.info(f"Saved feature importance plot to {importance_plot_path}")
        
        # Analyze the relationship between predictions and continuous mood score if available
        if mood_score_test is not None:
            plt.figure(figsize=(10, 6))
            plt.scatter(mood_score_test, y_prob, alpha=0.5)
            plt.xlabel('Actual Mood Score (0-10)')
            plt.ylabel('Predicted Probability of Good Mood')
            plt.title('Relationship Between Mood Score and Prediction')
            plt.grid(True, alpha=0.3)
            
            # Add best fit line
            z = np.polyfit(mood_score_test, y_prob, 1)
            p = np.poly1d(z)
            plt.plot(np.sort(mood_score_test), p(np.sort(mood_score_test)), "r--", alpha=0.8)
            
            # Save plot
            mood_plot_path = ENHANCED_DIR / f"{self.base_model_name}_mood_correlation.png"
            plt.savefig(mood_plot_path)
            logger.info(f"Saved mood correlation plot to {mood_plot_path}")
        
        # Save enhanced model and metadata
        metadata = {
            'model_type': type(model).__name__,
            'features': feature_names,
            'metrics': metrics,
            'additional_factors_used': [f for f in ADDITIONAL_FACTORS if f in X_train.columns],
            'trained_at': pd.Timestamp.now().isoformat()
        }
        
        self.enhanced_model = model
        self.enhanced_metadata = metadata
        
        joblib.dump(model, self.enhanced_model_path)
        joblib.dump(metadata, self.enhanced_metadata_path)
        
        logger.info(f"Saved enhanced model to {self.enhanced_model_path}")
        
        return model, metadata
    
    def predict_mood(self, sleep_metrics, additional_factors=None):
        # Predicts mood based on sleep metrics and additional lifestyle factors
        if self.enhanced_model is None:
            logger.error("Enhanced model not available")
            return None
        
        try:
            # Combine sleep metrics and additional factors
            features = sleep_metrics.copy()
            if additional_factors:
                features.update(additional_factors)
            
            # Construct DataFrame for prediction
            prediction_data = pd.DataFrame([features])
            
            # Get feature list for enhanced model
            enhanced_features = []
            if self.enhanced_metadata and 'features' in self.enhanced_metadata:
                enhanced_features = self.enhanced_metadata['features']
            elif hasattr(self.enhanced_model, 'feature_names_in_'):
                enhanced_features = list(self.enhanced_model.feature_names_in_)
            
            if not enhanced_features:
                logger.error("No feature information available for enhanced model")
                return None
                
            # Create a DataFrame with the correct features for the enhanced model
            enhanced_X = pd.DataFrame(index=[0])  # Initialize with one row
            for feature in enhanced_features:
                if feature in prediction_data.columns:
                    enhanced_X[feature] = prediction_data[feature].values
                else:
                    # Set missing feature to 0
                    enhanced_X[feature] = 0
            
            # Make sure DataFrame has the right shape (1 row)
            if enhanced_X.shape[0] != 1:
                logger.error(f"Enhanced feature DataFrame has wrong shape: {enhanced_X.shape}")
                return None
            
            # Make prediction
            try:
                proba = self.enhanced_model.predict_proba(enhanced_X)
                if proba.shape[0] == 0:
                    logger.error("Prediction produced empty probability array")
                    return None
                
                mood_probability = float(proba[0, 1])
                mood_prediction = bool(self.enhanced_model.predict(enhanced_X)[0])
                mood_score = mood_probability * 10  # Scale to 0-10
            except Exception as e:
                logger.error(f"Error making enhanced prediction: {e}")
                return None
            
            result = {
                'good_mood': mood_prediction,
                'probability': mood_probability,
                'mood_score': mood_score,
                'confidence': 'high' if mood_probability > 0.8 or mood_probability < 0.2 else 'medium'
            }
            
            # Compare with base model if available
            if self.base_model is not None:
                # Get feature list for base model
                base_features = []
                if self.base_metadata and 'features' in self.base_metadata:
                    base_features = self.base_metadata['features']
                elif self.base_metadata and 'selected_features' in self.base_metadata:
                    base_features = self.base_metadata['selected_features']
                elif hasattr(self.base_model, 'feature_names_in_'):
                    base_features = list(self.base_model.feature_names_in_)
                
                if base_features:
                    # Create a DataFrame with the correct features for the base model
                    base_X = pd.DataFrame(index=[0])  # Initialize with one row
                    for feature in base_features:
                        if feature in prediction_data.columns:
                            base_X[feature] = prediction_data[feature].values
                        else:
                            # Set missing feature to 0
                            base_X[feature] = 0
                    
                    # Make sure DataFrame has the right shape (1 row)
                    if base_X.shape[0] != 1:
                        logger.error(f"Base feature DataFrame has wrong shape: {base_X.shape}")
                        # Skip base model comparison
                        return result
                    
                    # Make base model prediction
                    try:
                        base_proba = self.base_model.predict_proba(base_X)
                        if base_proba.shape[0] == 0:
                            logger.error("Base prediction produced empty probability array")
                            return result
                        
                        base_probability = float(base_proba[0, 1])
                        base_prediction = bool(self.base_model.predict(base_X)[0])
                    except Exception as e:
                        logger.error(f"Error making base prediction: {e}")
                        return result
                    
                    # Add to result
                    result['base_prediction'] = {
                        'good_mood': base_prediction,
                        'probability': base_probability,
                        'mood_score': base_probability * 10
                    }
                    
                    # Calculate improvement
                    result['improvement'] = {
                        'absolute': float(abs(mood_probability - base_probability)),
                        'relative': float((mood_probability - base_probability) / base_probability if base_probability > 0 else 0),
                        'direction': 'higher' if mood_probability > base_probability else 'lower',
                        'prediction_changed': mood_prediction != base_prediction
                    }
            
            return result
        
        except Exception as e:
            logger.error(f"Error predicting mood: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def compare_with_base_model(self, n_samples=100):
        # Compares the enhanced model with the base model using synthetic test data
        if self.enhanced_model is None or self.base_model is None:
            logger.error("Both enhanced and base models are required for comparison")
            return None
        
        logger.info(f"Comparing enhanced model with base model using {n_samples} synthetic samples")
        
        # Generate synthetic test data
        np.random.seed(42)  # For reproducibility
        
        # Extract feature lists from both models
        base_features = []
        enhanced_features = []
        
        if self.base_metadata and 'features' in self.base_metadata:
            base_features = self.base_metadata['features']
        elif self.base_metadata and 'selected_features' in self.base_metadata:
            base_features = self.base_metadata['selected_features']
        
        if self.enhanced_metadata and 'features' in self.enhanced_metadata:
            enhanced_features = self.enhanced_metadata['features']
        
        if not base_features or not enhanced_features:
            logger.error("Unable to determine features for comparison")
            return None
        
        # Determine all needed features (union of both feature sets)
        all_features = list(set(base_features) | set(enhanced_features))
        logger.info(f"Using {len(all_features)} features for comparison")
        
        # Sleep metrics ranges based on typical values
        sleep_ranges = {
            'total_sleep_time': (4, 10),       # Hours
            'wake_time': (0, 120),             # Minutes
            'rem_time': (0, 3),                # Hours
            'light_sleep_time': (2, 6),        # Hours
            'deep_sleep_time': (0.5, 2.5),     # Hours
            'sleep_efficiency': (0.6, 1.0),    # Ratio
            'rem_percentage': (0.1, 0.35),     # Ratio
            'rem_cycles': (2, 6),              # Count
            'rem_awakenings': (0, 10)          # Count
        }
        
        # Additional factors ranges
        factor_ranges = {
            'stress_level': (0, 10),           # 0-10 scale
            'exercise_minutes': (0, 120),      # Minutes
            'caffeine_mg': (0, 500),           # Milligrams
            'screen_time_minutes': (0, 240),   # Minutes
            'alcohol_units': (0, 5),           # Units
            'outdoor_time_minutes': (0, 240),  # Minutes
            'social_interaction_score': (0, 10), # 0-10 scale
            'meditation_minutes': (0, 60)      # Minutes
        }
        
        # Generate samples with all possible features
        samples = []
        for _ in range(n_samples):
            # Generate all possible features
            sample = {}
            
            # Sleep metrics
            for feature, (min_val, max_val) in sleep_ranges.items():
                sample[feature] = np.random.uniform(min_val, max_val)
            
            # Additional factors
            for factor, (min_val, max_val) in factor_ranges.items():
                sample[factor] = np.random.uniform(min_val, max_val)
            
            samples.append(sample)
        
        # Create DataFrame with all features
        X_all = pd.DataFrame(samples)
        
        # Get predictions from both models
        base_results = []
        enhanced_results = []
        
        valid_samples = 0
        for i, row in X_all.iterrows():
            try:
                # Extract only needed features for each model
                features_dict = row.to_dict()
                
                # Base model prediction
                base_features_dict = {k: features_dict.get(k, 0) for k in base_features}
                base_df = pd.DataFrame([base_features_dict])
                base_prob = self.base_model.predict_proba(base_df)[0, 1]
                base_pred = bool(self.base_model.predict(base_df)[0])
                
                base_result = {
                    'good_mood': base_pred,
                    'probability': float(base_prob),
                    'mood_score': float(base_prob * 10)
                }
                
                # Enhanced model prediction
                enhanced_features_dict = {k: features_dict.get(k, 0) for k in enhanced_features}
                enhanced_df = pd.DataFrame([enhanced_features_dict])
                enhanced_prob = self.enhanced_model.predict_proba(enhanced_df)[0, 1]
                enhanced_pred = bool(self.enhanced_model.predict(enhanced_df)[0])
                
                enhanced_result = {
                    'good_mood': enhanced_pred,
                    'probability': float(enhanced_prob),
                    'mood_score': float(enhanced_prob * 10)
                }
                
                base_results.append(base_result)
                enhanced_results.append(enhanced_result)
                valid_samples += 1
                
            except Exception as e:
                # Skip this sample if there's an error
                logger.debug(f"Skipping sample due to: {e}")
                continue
                
            # Limit to requested number of valid samples
            if valid_samples >= n_samples:
                break
        
        # If we have no valid results, return
        if len(base_results) == 0 or len(enhanced_results) == 0:
            logger.error(f"Could not generate any valid predictions: base={len(base_results)}, enhanced={len(enhanced_results)}")
            return None
        
        # Analyze differences
        if len(base_results) != len(enhanced_results):
            logger.warning(f"Different number of results: base={len(base_results)}, enhanced={len(enhanced_results)}")
        
        # Calculate metrics
        different_predictions = sum(b['good_mood'] != e['good_mood'] for b, e in zip(base_results, enhanced_results))
        probability_differences = [e['probability'] - b['probability'] for b, e in zip(base_results, enhanced_results)]
        
        # Handle empty results case
        if not probability_differences:
            logger.error("No valid probability differences to analyze")
            return None
            
        mean_probability_diff = np.mean(probability_differences)
        median_probability_diff = np.median(probability_differences)
        max_probability_diff = max(probability_differences)
        min_probability_diff = min(probability_differences)
        
        # Create comparison report
        comparison = {
            'samples': len(base_results),
            'different_predictions': different_predictions,
            'prediction_change_rate': float(different_predictions / len(base_results)),
            'probability_difference': {
                'mean': float(mean_probability_diff),
                'median': float(median_probability_diff),
                'max': float(max_probability_diff),
                'min': float(min_probability_diff),
                'std': float(np.std(probability_differences))
            }
        }
        
        # Report comparison
        logger.info("\nModel comparison results:")
        logger.info(f"  Samples: {comparison['samples']}")
        logger.info(f"  Different predictions: {comparison['different_predictions']} ({comparison['prediction_change_rate']*100:.1f}%)")
        logger.info(f"  Probability difference:")
        logger.info(f"    Mean: {comparison['probability_difference']['mean']:.4f}")
        logger.info(f"    Median: {comparison['probability_difference']['median']:.4f}")
        logger.info(f"    Max: {comparison['probability_difference']['max']:.4f}")
        logger.info(f"    Min: {comparison['probability_difference']['min']:.4f}")
        logger.info(f"    Std: {comparison['probability_difference']['std']:.4f}")
        
        # Plot probability differences
        plt.figure(figsize=(10, 6))
        plt.hist(probability_differences, bins=20, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Probability Difference (Enhanced - Base)')
        plt.ylabel('Count')
        plt.title('Distribution of Prediction Probability Differences')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        diff_plot_path = ENHANCED_DIR / f"{self.base_model_name}_probability_diff.png"
        plt.savefig(diff_plot_path)
        logger.info(f"Saved probability difference plot to {diff_plot_path}")
        
        return comparison
    
    def load_real_data(self):
        # Loads real-world data for training the enhanced model
        logger.info("Loading real dataset for enhanced mood prediction")
        
        data_file = DATA_DIR / "enhanced_mood_data.csv"
        
        if not data_file.exists():
            logger.warning(f"Real data file not found: {data_file}")
            logger.info("Generating synthetic dataset instead")
            return self.generate_synthetic_dataset()
        
        try:
            # Load CSV data
            data = pd.read_csv(data_file)
            logger.info(f"Loaded {len(data)} records from {data_file}")
            
            # Check required columns
            required_sleep_metrics = ['total_sleep_time', 'sleep_efficiency', 'rem_percentage']
            missing_sleep_metrics = [col for col in required_sleep_metrics if col not in data.columns]
            
            if missing_sleep_metrics:
                logger.error(f"Missing required sleep metrics: {missing_sleep_metrics}")
                return None
            
            # Check for mood label
            if 'good_mood' not in data.columns and 'mood_rating' not in data.columns:
                logger.error("Missing mood label in dataset")
                return None
            
            # Create 'good_mood' column if needed
            if 'good_mood' not in data.columns and 'mood_rating' in data.columns:
                # Assuming mood_rating is on a 0-10 scale, with 7+ indicating good mood
                data['good_mood'] = (data['mood_rating'] >= 7).astype(int)
                logger.info("Created 'good_mood' label from 'mood_rating'")
            
            # Split features and labels
            X = data.drop(['good_mood'], axis=1)
            y = data['good_mood']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'data': data
            }
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

    def generate_synthetic_dataset(self, n_samples=1000):
        # Generates a synthetic dataset for training when real data is not available
        logger.info(f"Generating synthetic dataset with {n_samples} samples")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Define feature ranges
        sleep_ranges = {
            'total_sleep_time': (4, 10),       # Hours
            'wake_time': (0, 120),             # Minutes
            'rem_time': (0, 3),                # Hours
            'light_sleep_time': (2, 6),        # Hours
            'deep_sleep_time': (0.5, 2.5),     # Hours
            'sleep_efficiency': (0.6, 1.0),    # Ratio
            'rem_percentage': (0.1, 0.35),     # Ratio
            'rem_cycles': (2, 6),              # Count
            'rem_awakenings': (0, 10)          # Count
        }
        
        # Additional factors
        factor_ranges = {
            'stress_level': (0, 10),          # 0-10 scale
            'exercise_minutes': (0, 120),     # Minutes
            'caffeine_mg': (0, 500),          # Milligrams
            'screen_time_minutes': (0, 240),  # Minutes
            'alcohol_units': (0, 5),          # Units
            'outdoor_time_minutes': (0, 240), # Minutes
            'social_interaction_score': (0, 10), # 0-10 scale
            'meditation_minutes': (0, 60)     # Minutes
        }
        
        # Generate samples
        samples = []
        for _ in range(n_samples):
            # Generate sleep metrics
            sample = {}
            for feature, (min_val, max_val) in sleep_ranges.items():
                sample[feature] = np.random.uniform(min_val, max_val)
            
            # Generate additional factors
            for feature, (min_val, max_val) in factor_ranges.items():
                sample[feature] = np.random.uniform(min_val, max_val)
            
            # Generate synthetic mood label based on a combination of factors
            # Good sleep quality factors
            good_sleep = (
                sample['total_sleep_time'] > 7.0 and 
                sample['sleep_efficiency'] > 0.85 and
                sample['rem_percentage'] > 0.2
            )
            
            # Positive lifestyle factors
            positive_lifestyle = (
                sample['stress_level'] < 5.0 and
                sample['exercise_minutes'] > 30.0 and
                sample['outdoor_time_minutes'] > 60.0
            )
            
            # Negative factors
            negative_factors = (
                sample['caffeine_mg'] > 300.0 or
                sample['screen_time_minutes'] > 120.0 or
                sample['alcohol_units'] > 2.0
            )
            
            # Determine mood with some randomness
            # 70% chance of good mood if sleep and lifestyle are good and negative factors are low
            # 30% chance of good mood if sleep is poor and lifestyle is poor
            # 50% chance in other cases
            if good_sleep and positive_lifestyle and not negative_factors:
                mood_probability = 0.7
            elif not good_sleep and not positive_lifestyle:
                mood_probability = 0.3
            else:
                mood_probability = 0.5
                
            # Add some random noise to make the problem more realistic
            mood_probability += np.random.uniform(-0.15, 0.15)
            mood_probability = max(0.1, min(0.9, mood_probability))  # Keep between 0.1 and 0.9
            
            # Generate binary mood label
            sample['good_mood'] = int(np.random.random() < mood_probability)
            
            # Generate continuous mood score (0-10)
            base_mood_score = mood_probability * 10
            sample['mood_score'] = max(0, min(10, base_mood_score + np.random.normal(0, 1)))
            
            samples.append(sample)
        
        # Create DataFrame
        data = pd.DataFrame(samples)
        
        # Split features and labels
        X = data.drop(['good_mood'], axis=1)
        y = data['good_mood']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        logger.info(f"Generated synthetic dataset with {len(data)} samples")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'data': data
        }

    def check_and_create_models(self):
        # Checks if necessary models exist and creates them if needed
        logger.info("Checking if models exist and creating them if needed")
        
        # Check if base model exists
        if self.base_model is None:
            logger.warning("Base model not found, creating a simple base model")
            data = self.generate_synthetic_dataset(n_samples=500)
            
            # Create a simple random forest model for base
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Select only sleep metrics for base model
            sleep_features = [
                'total_sleep_time', 'wake_time', 'rem_time', 
                'light_sleep_time', 'deep_sleep_time', 'sleep_efficiency', 
                'rem_percentage'
            ]
            
            # Filter features
            available_sleep_features = [f for f in sleep_features if f in data['X_train'].columns]
            X_train_sleep = data['X_train'][available_sleep_features]
            
            # Train model
            model.fit(X_train_sleep, data['y_train'])
            
            # Save base model and metadata
            base_model_path = MODELS_DIR / "mood_prediction"
            base_model_path.mkdir(parents=True, exist_ok=True)
            
            self.base_model = model
            self.base_metadata = {
                'model_type': 'RandomForestClassifier',
                'features': available_sleep_features,
                'created_at': pd.Timestamp.now().isoformat()
            }
            
            joblib.dump(model, base_model_path / f"{self.base_model_name}.joblib")
            joblib.dump(self.base_metadata, base_model_path / f"{self.base_model_name}_metadata.joblib")
            logger.info(f"Created and saved base model with {len(available_sleep_features)} sleep features")
        
        # Check if enhanced model exists
        if self.enhanced_model is None:
            logger.warning("Enhanced model not found, creating a new enhanced model")
            data = self.generate_synthetic_dataset(n_samples=500)
            
            # Train enhanced model
            model, metadata = self.train_enhanced_model(
                data['X_train'], data['y_train'],
                data['X_test'], data['y_test']
            )
            
            logger.info(f"Created and saved enhanced model with {len(metadata['features'])} features")
        
        return self.base_model is not None and self.enhanced_model is not None

def main():
    # Main function to demonstrate enhanced mood prediction with lifestyle factors
    logger.info("Enhanced Mood Prediction Demo")
    
    # Create predictor instance
    predictor = EnhancedMoodPredictor("xgboost")
    
    # Ensure models exist before proceeding
    if not (predictor.base_model and predictor.enhanced_model):
        logger.info("One or more models missing, checking and creating as needed")
        if not predictor.check_and_create_models():
            logger.error("Failed to create necessary models")
            return 1
    
    # Compare models
    logger.info("\nComparing base and enhanced models:")
    try:
        comparison = predictor.compare_with_base_model(n_samples=50)
        if comparison:
            logger.info(f"Prediction changes: {comparison['different_predictions']} out of {comparison['samples']} samples ({comparison['prediction_change_rate']*100:.1f}%)")
            logger.info(f"Mean probability difference: {comparison['probability_difference']['mean']:.4f}")
            if comparison['probability_difference']['mean'] > 0:
                logger.info("On average, the enhanced model predicts a higher probability of good mood")
            else:
                logger.info("On average, the enhanced model predicts a lower probability of good mood")
        else:
            logger.warning("Model comparison failed")
    except Exception as e:
        logger.error(f"Error during model comparison: {e}")
        logger.info("Continuing with example prediction...")
    
    # Example prediction
    logger.info("\nExample prediction:")
    
    # Example sleep metrics
    sleep_metrics = {
        'total_sleep_time': 7.5,
        'wake_time': 20,
        'rem_time': 1.8,
        'light_sleep_time': 4.0,
        'deep_sleep_time': 1.7,
        'sleep_efficiency': 0.92,
        'rem_percentage': 0.24
    }
    
    # Example additional factors
    additional_factors = {
        'stress_level': 4,
        'exercise_minutes': 45,
        'caffeine_mg': 150,
        'screen_time_minutes': 60,
        'alcohol_units': 1,
        'outdoor_time_minutes': 90,
        'social_interaction_score': 7,
        'meditation_minutes': 10
    }
    
    # Make prediction
    try:
        result = predictor.predict_mood(sleep_metrics, additional_factors)
        
        if result:
            logger.info(f"Enhanced prediction: {'Good' if result['good_mood'] else 'Bad'} mood (score: {result['mood_score']:.1f}/10)")
            
            if 'base_prediction' in result:
                logger.info(f"Base prediction: {'Good' if result['base_prediction']['good_mood'] else 'Bad'} mood (score: {result['base_prediction']['mood_score']:.1f}/10)")
                
                if 'improvement' in result and result['improvement']['prediction_changed']:
                    logger.info(f"The additional factors changed the prediction!")
                    logger.info(f"Absolute difference: {result['improvement']['absolute']:.4f}")
                    logger.info(f"Direction: {result['improvement']['direction']}")
        else:
            logger.error("Failed to get prediction")
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Generate additional example with different values
    logger.info("\nAdditional example prediction:")
    
    # Different sleep metrics (poor sleep quality)
    poor_sleep_metrics = {
        'total_sleep_time': 5.2,
        'wake_time': 75,
        'rem_time': 0.9,
        'light_sleep_time': 3.5,
        'deep_sleep_time': 0.8,
        'sleep_efficiency': 0.65,
        'rem_percentage': 0.17
    }
    
    # Different additional factors (good lifestyle)
    good_lifestyle_factors = {
        'stress_level': 2,
        'exercise_minutes': 90,
        'caffeine_mg': 50,
        'screen_time_minutes': 30,
        'alcohol_units': 0,
        'outdoor_time_minutes': 120,
        'social_interaction_score': 9,
        'meditation_minutes': 20
    }
    
    # Make prediction with poor sleep but good lifestyle
    try:
        result = predictor.predict_mood(poor_sleep_metrics, good_lifestyle_factors)
        
        if result:
            logger.info(f"Enhanced prediction: {'Good' if result['good_mood'] else 'Bad'} mood (score: {result['mood_score']:.1f}/10)")
            
            if 'base_prediction' in result:
                logger.info(f"Base prediction: {'Good' if result['base_prediction']['good_mood'] else 'Bad'} mood (score: {result['base_prediction']['mood_score']:.1f}/10)")
                
                if 'improvement' in result and result['improvement']['prediction_changed']:
                    logger.info(f"The additional factors changed the prediction!")
                    logger.info(f"This demonstrates how good lifestyle factors can potentially offset poor sleep quality.")
        else:
            logger.error("Failed to get prediction")
    except Exception as e:
        logger.error(f"Error during additional prediction: {e}")
    
    logger.info("\nEnhanced Mood Prediction completed")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 