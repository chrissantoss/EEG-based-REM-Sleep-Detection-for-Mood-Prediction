#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script enhances mood prediction by incorporating additional lifestyle factors
beyond sleep metrics, such as stress, exercise, diet, and caffeine consumption.
"""

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
    """Class for enhanced mood prediction with additional factors."""
    
    def __init__(self, base_model_name="xgboost"):
        """
        Initialize the enhanced mood predictor.
        
        Args:
            base_model_name (str): Name of the base model to enhance
        """
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
        """
        Load the base model for mood prediction.
        
        Returns:
            tuple: (model, metadata) or (None, None) if loading fails
        """
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
            
            logger.info(f"Loaded base model: {model_path}")
            return model, metadata
        
        except Exception as e:
            logger.error(f"Error loading base model: {e}")
            return None, None
        
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
            plt.title('Relationship Between Mood Score and Model Predictions')
            plt.grid(True)
            
            # Add best fit line
            z = np.polyfit(mood_score_test, y_prob, 1)
            p = np.poly1d(z)
            plt.plot(range(11), p(range(11)), "r--")
            
            # Save plot
            correlation_plot_path = ENHANCED_DIR / f"{self.base_model_name}_correlation.png"
            plt.savefig(correlation_plot_path)
            logger.info(f"Saved correlation plot to {correlation_plot_path}")
        
        # Create metadata
        metadata = {
            'model_name': f"{self.base_model_name}_enhanced",
            'metrics': metrics,
            'feature_importance': importance_df.to_dict() if hasattr(model, 'feature_importances_') else None,
            'includes_additional_factors': True,
            'additional_factors': ADDITIONAL_FACTORS,
            'description': 'Enhanced model incorporating additional lifestyle factors beyond sleep metrics'
        }
        
        # Save model and metadata
        joblib.dump(model, self.enhanced_model_path)
        joblib.dump(metadata, self.enhanced_metadata_path)
        
        logger.info(f"Saved enhanced model to {self.enhanced_model_path}")
        
        self.enhanced_model = model
        self.enhanced_metadata = metadata
        
        return model, metadata
    
    def predict_mood(self, sleep_metrics, additional_factors=None):
        """
        Predict mood based on sleep metrics and additional factors.
        
        Args:
            sleep_metrics (dict): Sleep metrics
            additional_factors (dict, optional): Additional lifestyle factors
            
        Returns:
            dict: Prediction results
        """
        # Check if enhanced model is available
        if self.enhanced_model is None:
            logger.error("Enhanced model not available. Train model first.")
            return None
        
        try:
            # Combine input data
            input_data = sleep_metrics.copy()
            
            # Add additional factors if provided
            if additional_factors:
                input_data.update(additional_factors)
            
            # Ensure all required factors are present
            missing_factors = []
            for factor in ADDITIONAL_FACTORS:
                if factor not in input_data:
                    input_data[factor] = 0  # Default to 0 if not provided
                    missing_factors.append(factor)
            
            # Get the features that the model was trained on
            if hasattr(self.enhanced_model, 'feature_names_in_'):
                # For newer scikit-learn versions
                expected_features = self.enhanced_model.feature_names_in_
            else:
                # Fallback to a basic list of features
                expected_features = [
                    'total_sleep_time', 'wake_time', 'rem_time', 
                    'light_sleep_time', 'deep_sleep_time', 'sleep_efficiency', 
                    'rem_percentage'
                ] + ADDITIONAL_FACTORS
                
            # Create DataFrame with all expected features
            X = pd.DataFrame([input_data])
            
            # Check for missing expected features and add them
            for feature in expected_features:
                if feature not in X.columns:
                    logger.warning(f"Adding missing feature for prediction: {feature}")
                    X[feature] = 0
                        
            # Ensure we only use the features the model expects
            if hasattr(self.enhanced_model, 'feature_names_in_'):
                X = X[self.enhanced_model.feature_names_in_]
                    
            # Make prediction
            prediction = bool(self.enhanced_model.predict(X)[0])
            probability = float(self.enhanced_model.predict_proba(X)[0, 1])
            
            # Create result
            result = {
                'good_mood': prediction,
                'probability': probability,
                'mood_score': probability * 10,  # Scale to 0-10
                'includes_additional_factors': True,
                'missing_factors': missing_factors
            }
            
            # Add feature contribution if available
            if hasattr(self.enhanced_model, 'feature_importances_'):
                # Multiply feature values by importance to get contribution
                feature_importance = {
                    feature: float(self.enhanced_model.feature_importances_[i])
                    for i, feature in enumerate(X.columns)
                }
                
                # Sort by absolute contribution
                result['feature_importance'] = {
                    k: v for k, v in sorted(
                        feature_importance.items(), 
                        key=lambda item: abs(item[1]), 
                        reverse=True
                    )
                }
            
            return result
        
        except Exception as e:
            logger.error(f"Error predicting mood: {e}")
            return None
    
    def compare_with_base_model(self, n_samples=100):
        """
        Compare enhanced model with base model on real data.
        
        Args:
            n_samples (int): Maximum number of samples to use for comparison
            
        Returns:
            dict: Comparison results
        """
        if self.base_model is None or self.enhanced_model is None:
            logger.error("Both base and enhanced models must be available for comparison")
            return None
        
        try:
            # Load real test data
            X, y = self.load_real_data()
            
            if X is None or y is None:
                logger.error("Failed to load real data for comparison")
                return None
                
            # Limit samples if needed
            if len(X) > n_samples:
                # Use stratified sampling to maintain class distribution
                from sklearn.model_selection import train_test_split
                _, X, _, y = train_test_split(X, y, test_size=n_samples/len(X), stratify=y, random_state=42)
                logger.info(f"Using {len(X)} samples for comparison")
                
            # Store mood score if available
            mood_score = None
            if 'mood_score' in X.columns:
                mood_score = X['mood_score'].copy()
                X = X.drop('mood_score', axis=1)
            
            # Prepare data for base model - handle feature mismatch
            X_base = X.copy()
            
            # Check if we need to add missing features for the base model
            if self.base_metadata and 'selected_features' in self.base_metadata:
                selected_features = self.base_metadata['selected_features']
                missing_features = [f for f in selected_features if f not in X_base.columns]
                
                # Add missing features with zero values
                for feature in missing_features:
                    logger.warning(f"Adding missing feature for base model: {feature}")
                    X_base[feature] = 0
                    
                # Only use the selected features
                X_base = X_base[selected_features]
            
            # Check for base model features that don't exist in real data
            expected_base_features = list(X_base.columns)
            
            logger.info(f"Features used for base model: {expected_base_features}")
            
            # Add any missing features for the enhanced model
            expected_enhanced_features = list(X.columns)
            for feature in ADDITIONAL_FACTORS:
                if feature not in X.columns:
                    logger.warning(f"Adding missing additional factor: {feature}")
                    X[feature] = 0
            
            logger.info(f"Features used for enhanced model: {list(X.columns)}")
            
            # Make predictions
            base_pred = self.base_model.predict(X_base)
            base_prob = self.base_model.predict_proba(X_base)[:, 1]
            
            enhanced_pred = self.enhanced_model.predict(X)
            enhanced_prob = self.enhanced_model.predict_proba(X)[:, 1]
            
            # Calculate metrics
            base_metrics = {
                'accuracy': float(accuracy_score(y, base_pred)),
                'precision': float(precision_score(y, base_pred)),
                'recall': float(recall_score(y, base_pred)),
                'f1': float(f1_score(y, base_pred)),
                'roc_auc': float(roc_auc_score(y, base_prob))
            }
            
            enhanced_metrics = {
                'accuracy': float(accuracy_score(y, enhanced_pred)),
                'precision': float(precision_score(y, enhanced_pred)),
                'recall': float(recall_score(y, enhanced_pred)),
                'f1': float(f1_score(y, enhanced_pred)),
                'roc_auc': float(roc_auc_score(y, enhanced_prob))
            }
            
            # Calculate differences
            diff_metrics = {
                metric: enhanced_metrics[metric] - base_metrics[metric]
                for metric in base_metrics
            }
            
            # Create comparison result
            comparison = {
                'base_model': {
                    'name': self.base_model_name,
                    'metrics': base_metrics
                },
                'enhanced_model': {
                    'name': f"{self.base_model_name}_enhanced",
                    'metrics': enhanced_metrics
                },
                'differences': diff_metrics,
                'sample_size': len(X)
            }
            
            # Log results
            logger.info("\nModel Comparison Results:")
            logger.info(f"Base Model ({self.base_model_name}):")
            for metric, value in base_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            logger.info(f"\nEnhanced Model ({self.base_model_name}_enhanced):")
            for metric, value in enhanced_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            logger.info("\nImprovement:")
            for metric, diff in diff_metrics.items():
                sign = '+' if diff >= 0 else ''
                logger.info(f"  {metric}: {sign}{diff:.4f} ({diff/base_metrics[metric]*100:.1f}%)")
            
            # Visualize comparison
            plt.figure(figsize=(12, 6))
            metrics = list(base_metrics.keys())
            base_values = [base_metrics[m] for m in metrics]
            enhanced_values = [enhanced_metrics[m] for m in metrics]
            
            x = range(len(metrics))
            width = 0.35
            
            plt.bar([i - width/2 for i in x], base_values, width, label='Base Model')
            plt.bar([i + width/2 for i in x], enhanced_values, width, label='Enhanced Model')
            
            plt.xlabel('Metric')
            plt.ylabel('Value')
            plt.title('Model Performance Comparison')
            plt.xticks(x, metrics)
            plt.legend()
            plt.grid(True, axis='y')
            
            # Save plot
            comparison_plot_path = ENHANCED_DIR / f"{self.base_model_name}_comparison.png"
            plt.savefig(comparison_plot_path)
            logger.info(f"Saved comparison plot to {comparison_plot_path}")
            
            return comparison
        
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return None

    def load_real_data(self):
        """
        Load real sleep mood data from the dataset.
        
        Returns:
            tuple: (X, y) - features and labels
        """
        logger.info("Loading real sleep mood data")
        
        # Define paths to real data
        processed_data_path = DATA_DIR / "processed" / "sleep_efficiency_research_based.csv"
        # Use only the processed dataset
        
        try:
            # Try to load processed data first
            if processed_data_path.exists():
                df = pd.read_csv(processed_data_path)
                logger.info(f"Loaded processed data with {len(df)} records from {processed_data_path}")
            # Fall back to raw data if necessary
            elif raw_data_path.exists():
                df = pd.read_csv(raw_data_path)
                logger.info(f"Loaded raw data with {len(df)} records from {raw_data_path}")
            else:
                logger.error("No real data found. Please ensure data files exist.")
                return None, None
            
            # For the sleep_efficiency_research_based.csv dataset
            if 'good_mood' in df.columns:
                # Features include sleep metrics and additional factors if available
                sleep_features = ['total_sleep_time', 'wake_time', 'rem_time', 
                                'light_sleep_time', 'deep_sleep_time', 'sleep_efficiency', 
                                'rem_percentage']
                
                additional_features = []
                for factor in ADDITIONAL_FACTORS:
                    if factor in df.columns:
                        additional_features.append(factor)
                
                X = df[sleep_features + additional_features]
                y = df['good_mood']
                
            # Handle unsupported data formats
            else:
                logger.error("Unsupported data format. Missing expected sleep efficiency columns.")
                return None, None
            
            # Handle missing values if any
            X = X.fillna(X.mean())
            
            logger.info(f"Prepared dataset with {len(X)} records and {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading real data: {e}")
            return None, None

def main():
    """Main function to demonstrate enhanced mood prediction."""
    logger.info("Enhancing mood prediction with additional factors")
    
    # Create enhanced predictor
    predictor = EnhancedMoodPredictor("xgboost")
    
    # IMPORTANT: Use real data for mood prediction, NOT synthetic data
    X, y = predictor.load_real_data()
    
    # Check if data was loaded successfully
    if X is None or y is None:
        logger.error("Failed to load real data. Cannot proceed with training.")
        return 1
    
    # Train enhanced model
    model, metadata = predictor.train_enhanced_model(X, y)
    
    # Compare with base model
    comparison = predictor.compare_with_base_model(n_samples=100)
    
    # Demonstrate prediction
    test_sleep_metrics = {
        'total_sleep_time': 450,  # 7.5 hours
        'wake_time': 20,
        'rem_time': 100,
        'light_sleep_time': 240,
        'deep_sleep_time': 90,
        'sleep_efficiency': 95,
        'rem_percentage': 22,
        'rem_cycles': 4,
        'rem_awakenings': 2
    }
    
    test_additional_factors = {
        'stress_level': 3,            # Low stress (1-10)
        'exercise_minutes': 45,        # 45 minutes of exercise
        'caffeine_mg': 160,            # 2 cups of coffee
        'screen_time_minutes': 30,     # 30 minutes before bed
        'alcohol_units': 1,            # 1 alcoholic drink
        'outdoor_time_minutes': 60,    # 1 hour outdoors
        'social_interaction_score': 7, # Good social day
        'meditation_minutes': 15       # 15 minutes of meditation
    }
    
    # Make prediction
    result = predictor.predict_mood(test_sleep_metrics, test_additional_factors)
    
    if result:
        logger.info("\nMood Prediction for Test Input:")
        logger.info(f"  Predicted mood: {'Good' if result['good_mood'] else 'Bad'}")
        logger.info(f"  Confidence: {result['probability']:.2f}")
        logger.info(f"  Mood score (0-10): {result['mood_score']:.1f}")
        
        if 'feature_importance' in result:
            logger.info("\nTop 5 factors influencing this prediction:")
            for i, (feature, importance) in enumerate(list(result['feature_importance'].items())[:5]):
                logger.info(f"  {i+1}. {feature}: {importance:.4f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 