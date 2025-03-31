#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Implementation of a personalized mood prediction system that calibrates based on individual sleep patterns.

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Define directories
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
INDIVIDUAL_DIR = ROOT_DIR / "individual_data"
INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)

class PersonalizedMoodPredictor:
    # Handles personalized mood prediction with calibration based on user data
    
    def __init__(self, user_id, base_model_name="xgboost_refined"):
        # Initialize the personalized mood predictor with user ID and base model
        self.user_id = user_id
        self.base_model_name = base_model_name
        self.user_dir = INDIVIDUAL_DIR / user_id
        self.user_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths to user data and model
        self.sleep_data_path = self.user_dir / "sleep_data.csv"
        self.personal_model_path = self.user_dir / f"personal_model.joblib"
        self.calibration_path = self.user_dir / "calibration_data.joblib"
        
        # Load base model
        self.base_model, self.base_metadata = self.load_base_model()
        
        # Initialize personal data
        self.personal_data = self.load_personal_data()
        
        # Initialize calibration model
        self.calibration_model = self.load_calibration_model()
    
    def load_base_model(self):
        # Loads the base model for mood prediction, returning (model, metadata) or (None, None) if loading fails
        model_path = MODELS_DIR / "mood_prediction" / f"{self.base_model_name}.joblib"
        metadata_path = MODELS_DIR / "mood_prediction" / f"{self.base_model_name}_metadata.joblib"
        
        if not model_path.exists() or not metadata_path.exists():
            # Try without _refined suffix
            base_name = self.base_model_name.replace("_refined", "")
            model_path = MODELS_DIR / "mood_prediction" / f"{base_name}.joblib"
            metadata_path = MODELS_DIR / "mood_prediction" / f"{base_name}_metadata.joblib"
            
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
    
    def load_personal_data(self):
        # Loads personal sleep and mood data, creating empty DataFrame if none exists
        if not self.sleep_data_path.exists():
            # Create an empty DataFrame
            return pd.DataFrame(columns=[
                'date', 'total_sleep_time', 'wake_time', 'rem_time', 
                'light_sleep_time', 'deep_sleep_time', 'sleep_efficiency', 
                'rem_percentage', 'rem_cycles', 'rem_awakenings', 'mood_rating',
                'stress_level', 'exercise_minutes', 'caffeine_mg'
            ])
        
        try:
            # Load existing data
            data = pd.read_csv(self.sleep_data_path)
            logger.info(f"Loaded personal data: {len(data)} records")
            return data
        
        except Exception as e:
            logger.error(f"Error loading personal data: {e}")
            return pd.DataFrame()
    
    def load_calibration_model(self):
        # Loads the personal calibration model if available
        if not self.calibration_path.exists():
            logger.info("No calibration model found")
            return None
        
        try:
            calibration_data = joblib.load(self.calibration_path)
            logger.info("Loaded calibration model")
            return calibration_data
        
        except Exception as e:
            logger.error(f"Error loading calibration model: {e}")
            return None
    
    def add_sleep_mood_record(self, sleep_metrics, mood_rating, 
                             stress_level=None, exercise_minutes=None, 
                             caffeine_mg=None):
        # Adds a new sleep and mood record to the user's data and updates calibration if enough data
        try:
            # Create a new record
            record = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'mood_rating': mood_rating,
                'stress_level': stress_level,
                'exercise_minutes': exercise_minutes,
                'caffeine_mg': caffeine_mg
            }
            
            # Add sleep metrics
            record.update(sleep_metrics)
            
            # Add to DataFrame using concat instead of append
            new_record_df = pd.DataFrame([record])
            self.personal_data = pd.concat([self.personal_data, new_record_df], ignore_index=True)
            
            # Save to file
            self.personal_data.to_csv(self.sleep_data_path, index=False)
            
            logger.info(f"Added new sleep efficiency and mood record for {record['date']}")
            
            # Update calibration model if we have enough data
            if len(self.personal_data) >= 5:
                self.update_calibration_model()
            
            return True
        
        except Exception as e:
            logger.error(f"Error adding sleep efficiency and mood record: {e}")
            return False
    
    def update_calibration_model(self):
        # Updates the personal calibration model using collected data, needs at least 5 records
        if len(self.personal_data) < 5:
            logger.warning("Not enough data to update calibration model (min 5 records)")
            return False
        
        try:
            # Prepare data
            X = self.personal_data.drop(['date', 'mood_rating'], axis=1, errors='ignore')
            
            # Handle missing values
            X = X.fillna(0)
            
            # Get mood ratings
            y = (self.personal_data['mood_rating'] >= 7).astype(int)  # Consider >= 7 as "good mood"
            
            # Get base model predictions
            if self.base_model is not None:
                # Get required features for base model
                if self.base_metadata and 'selected_features' in self.base_metadata:
                    required_features = self.base_metadata['selected_features']
                    missing_features = [f for f in required_features if f not in X.columns]
                    
                    if missing_features:
                        logger.warning(f"Missing features for base model: {missing_features}")
                        # Add missing features with zeros
                        for feature in missing_features:
                            X[feature] = 0
                    
                    base_X = X[required_features]
                else:
                    base_X = X
                
                base_predictions = self.base_model.predict(base_X)
                base_probas = self.base_model.predict_proba(base_X)[:, 1]
                
                # Add base model predictions as features
                X['base_prediction'] = base_predictions
                X['base_confidence'] = base_probas
            
            # Split data for calibration
            if len(X) >= 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
            else:
                X_train, y_train = X, y
                X_test, y_test = None, None
            
            # Train calibration model
            calibration = LogisticRegression(random_state=42)
            calibration.fit(X_train, y_train)
            
            # Evaluate if possible
            if X_test is not None and len(X_test) > 0:
                # Calculate metrics
                y_pred = calibration.predict(X_test)
                y_prob = calibration.predict_proba(X_test)[:, 1]
                
                metrics = {
                    'accuracy': float(accuracy_score(y_test, y_pred)),
                    'f1': float(f1_score(y_test, y_pred))
                }
                
                logger.info("Calibration model performance:")
                for metric, value in metrics.items():
                    logger.info(f"  {metric}: {value:.4f}")
            
            # Save calibration model
            calibration_data = {
                'model': calibration,
                'features': X.columns.tolist(),
                'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            joblib.dump(calibration_data, self.calibration_path)
            logger.info(f"Saved calibration model to {self.calibration_path}")
            
            self.calibration_model = calibration_data
            
            return True
        
        except Exception as e:
            logger.error(f"Error updating calibration model: {e}")
            return False
    
    def predict_mood(self, sleep_metrics, additional_factors=None):
        # Predicts mood based on sleep metrics and additional factors, applying personal calibration
        # Returns tuple of (prediction, confidence)
        try:
            # Combine metrics and factors
            features = sleep_metrics.copy()
            
            if additional_factors is not None:
                features.update(additional_factors)
            
            # Convert to DataFrame
            X = pd.DataFrame([features])
            
            # Base prediction
            base_prediction = None
            base_confidence = 0.5
            
            if self.base_model is not None:
                # Check if we have all required features
                if self.base_metadata and 'selected_features' in self.base_metadata:
                    required_features = self.base_metadata['selected_features']
                    missing_features = [f for f in required_features if f not in X.columns]
                    
                    if missing_features:
                        logger.warning(f"Missing features for base model: {missing_features}")
                        # Add missing features with zeros
                        for feature in missing_features:
                            X[feature] = 0
                    
                    base_X = X[required_features]
                else:
                    base_X = X
                
                # Get base model prediction
                base_prediction = bool(self.base_model.predict(base_X)[0])
                base_confidence = float(self.base_model.predict_proba(base_X)[0, 1])
                
                logger.info(f"Base model prediction: {base_prediction} (confidence: {base_confidence:.4f})")
            
            # Apply calibration if available
            if self.calibration_model is not None:
                calibration = self.calibration_model['model']
                required_features = self.calibration_model['features']
                
                # Prepare features for calibration
                if 'base_prediction' in required_features and base_prediction is not None:
                    X['base_prediction'] = int(base_prediction)
                
                if 'base_confidence' in required_features and base_prediction is not None:
                    X['base_confidence'] = base_confidence
                
                # Check missing features
                missing_features = [f for f in required_features if f not in X.columns]
                if missing_features:
                    logger.warning(f"Missing features for calibration: {missing_features}")
                    # Add missing features with zeros
                    for feature in missing_features:
                        X[feature] = 0
                
                # Select required features
                X_cal = X[required_features]
                
                # Get calibrated prediction
                calibrated_prediction = bool(calibration.predict(X_cal)[0])
                calibrated_confidence = float(calibration.predict_proba(X_cal)[0, 1])
                
                logger.info(f"Calibrated prediction: {calibrated_prediction} (confidence: {calibrated_confidence:.4f})")
                
                return calibrated_prediction, calibrated_confidence
            
            # Fallback to base prediction
            return base_prediction, base_confidence
        
        except Exception as e:
            logger.error(f"Error predicting mood: {e}")
            return None, 0.0

def simulate_personalization():
    # Simulates personalization process with synthetic data to demonstrate calibration
    logger.info("Simulating personalization process...")
    
    # Create a sample user
    user_id = f"sim_user_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    predictor = PersonalizedMoodPredictor(user_id)
    
    # Generate some synthetic data - imagine this user has better mood with less sleep
    # than the average population
    np.random.seed(42)
    
    # Let's generate 20 days of data
    for i in range(20):
        # Generate sleep metrics
        sleep_time = np.random.normal(6.5, 0.5)  # Mean sleep time of 6.5 hours
        rem_time = sleep_time * np.random.normal(0.25, 0.05)  # About 25% REM
        deep_time = sleep_time * np.random.normal(0.20, 0.05)  # About 20% deep sleep
        light_time = sleep_time - rem_time - deep_time
        
        wake_time = np.random.normal(20, 10)  # Minutes awake during the night
        rem_cycles = np.random.normal(4, 1)  # Number of REM cycles
        rem_awakenings = np.random.normal(2, 1)  # Number of awakenings from REM
        
        # For this user, shorter sleep actually leads to better mood (unusual)
        base_mood = 10 - sleep_time  # Invert the normal relationship
        
        # Add some noise
        mood_with_noise = base_mood + np.random.normal(0, 1)
        mood_rating = max(1, min(10, mood_with_noise))  # Clamp between 1-10
        
        # Create sleep metrics
        sleep_metrics = {
            'total_sleep_time': sleep_time,
            'wake_time': wake_time,
            'rem_time': rem_time,
            'light_sleep_time': light_time,
            'deep_sleep_time': deep_time,
            'sleep_efficiency': (sleep_time * 60 - wake_time) / (sleep_time * 60),
            'rem_percentage': rem_time / sleep_time,
            'rem_cycles': rem_cycles,
            'rem_awakenings': rem_awakenings
        }
        
        # Additional factors
        additional_factors = {
            'stress_level': np.random.normal(4, 2),
            'exercise_minutes': np.random.normal(30, 15),
            'caffeine_mg': np.random.normal(150, 50)
        }
        
        # Add the record
        predictor.add_sleep_mood_record(
            sleep_metrics, mood_rating,
            stress_level=additional_factors['stress_level'],
            exercise_minutes=additional_factors['exercise_minutes'],
            caffeine_mg=additional_factors['caffeine_mg']
        )
        
        # After 10 days, let's see the predictions
        if i == 9:
            logger.info("\n=== After 10 days of data ===")
            
            # Test with different sleep durations
            for test_sleep in [5.0, 6.0, 7.0, 8.0]:
                # Create test sleep metrics
                test_metrics = {
                    'total_sleep_time': test_sleep,
                    'wake_time': 20,
                    'rem_time': test_sleep * 0.25,
                    'light_sleep_time': test_sleep * 0.55,
                    'deep_sleep_time': test_sleep * 0.20,
                    'sleep_efficiency': (test_sleep * 60 - 20) / (test_sleep * 60),
                    'rem_percentage': 0.25,
                    'rem_cycles': 4,
                    'rem_awakenings': 2
                }
                
                # Make prediction
                prediction, confidence = predictor.predict_mood(test_metrics, additional_factors)
                expected_mood = 10 - test_sleep  # The unusual pattern for this user
                
                logger.info(f"Sleep: {test_sleep}h → Prediction: {'Good' if prediction else 'Bad'} " +
                          f"(conf: {confidence:.4f}, expected mood: {expected_mood:.1f})")
    
    logger.info("\n=== After 20 days of data ===")
    
    # Test again with different sleep durations
    for test_sleep in [5.0, 6.0, 7.0, 8.0]:
        # Create test sleep metrics
        test_metrics = {
            'total_sleep_time': test_sleep,
            'wake_time': 20,
            'rem_time': test_sleep * 0.25,
            'light_sleep_time': test_sleep * 0.55,
            'deep_sleep_time': test_sleep * 0.20,
            'sleep_efficiency': (test_sleep * 60 - 20) / (test_sleep * 60),
            'rem_percentage': 0.25,
            'rem_cycles': 4,
            'rem_awakenings': 2
        }
        
        # Make prediction
        prediction, confidence = predictor.predict_mood(test_metrics, additional_factors)
        expected_mood = 10 - test_sleep  # The unusual pattern for this user
        
        logger.info(f"Sleep: {test_sleep}h → Prediction: {'Good' if prediction else 'Bad'} " +
                   f"(conf: {confidence:.4f}, expected mood: {expected_mood:.1f})")

def main():
    # Main function to demonstrate the PersonalizedMoodPredictor
    logger.info("Personalized Mood Predictor Demo")
    
    # Create a predictor instance
    user_id = "demo_user"
    predictor = PersonalizedMoodPredictor(user_id)
    
    # Check if we have enough data already
    if len(predictor.personal_data) < 5:
        logger.info("Not enough personal data, simulating personalization...")
        simulate_personalization()
    else:
        logger.info(f"Found {len(predictor.personal_data)} records for user {user_id}")
        
        # Test current model with some sleep values
        for sleep_hours in [6.0, 7.0, 8.0, 9.0]:
            test_metrics = {
                'total_sleep_time': sleep_hours,
                'wake_time': 20,
                'rem_time': sleep_hours * 0.25,
                'light_sleep_time': sleep_hours * 0.55,
                'deep_sleep_time': sleep_hours * 0.20,
                'sleep_efficiency': (sleep_hours * 60 - 20) / (sleep_hours * 60),
                'rem_percentage': 0.25,
                'rem_cycles': 4,
                'rem_awakenings': 2
            }
            
            # Make prediction
            prediction, confidence = predictor.predict_mood(test_metrics)
            
            logger.info(f"Sleep: {sleep_hours}h → Mood prediction: {'Good' if prediction else 'Bad'} " +
                       f"(confidence: {confidence:.4f})")

if __name__ == "__main__":
    main() 