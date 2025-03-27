#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script implements a personalized mood prediction system 
that calibrates based on individual sleep patterns.
"""

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
    """Class for personalized mood prediction with calibration."""
    
    def __init__(self, user_id, base_model_name="xgboost_refined"):
        """
        Initialize the personalized mood predictor.
        
        Args:
            user_id (str): Unique identifier for the user
            base_model_name (str): Name of the base model to use
        """
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
        """
        Load the base model for mood prediction.
        
        Returns:
            tuple: (model, metadata) or (None, None) if loading fails
        """
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
        """
        Load personal sleep and mood data.
        
        Returns:
            pd.DataFrame: DataFrame containing sleep and mood data
        """
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
        """
        Load the personal calibration model.
        
        Returns:
            object: Calibration model or None if not available
        """
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
        """
        Add a new sleep and mood record to the user's data.
        
        Args:
            sleep_metrics (dict): Sleep metrics
            mood_rating (float): User's self-reported mood rating (0-10)
            stress_level (float, optional): Stress level (0-10)
            exercise_minutes (int, optional): Exercise duration in minutes
            caffeine_mg (int, optional): Caffeine consumption in mg
            
        Returns:
            bool: True if record was added successfully
        """
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
        """
        Update the personal calibration model using collected data.
        
        Returns:
            bool: True if model was updated successfully
        """
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
                y_pred = calibration.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                logger.info(f"Calibration model performance: Accuracy={accuracy:.4f}, F1={f1:.4f}")
                
                # Store evaluation metrics
                calibration_data = {
                    'model': calibration,
                    'metrics': {
                        'accuracy': float(accuracy),
                        'f1': float(f1)
                    },
                    'features': list(X.columns),
                    'updated_at': datetime.now().isoformat()
                }
            else:
                calibration_data = {
                    'model': calibration,
                    'metrics': {},
                    'features': list(X.columns),
                    'updated_at': datetime.now().isoformat()
                }
            
            # Save calibration model
            joblib.dump(calibration_data, self.calibration_path)
            self.calibration_model = calibration_data
            
            logger.info(f"Updated calibration model with {len(X_train)} records")
            return True
        
        except Exception as e:
            logger.error(f"Error updating calibration model: {e}")
            return False
    
    def predict_mood(self, sleep_metrics, additional_factors=None):
        """
        Predict mood based on sleep metrics with personal calibration.
        
        Args:
            sleep_metrics (dict): Sleep metrics
            additional_factors (dict, optional): Additional factors like stress, exercise, etc.
            
        Returns:
            dict: Prediction results
        """
        if self.base_model is None:
            logger.error("Base model not available")
            return None
        
        try:
            # Combine sleep metrics and additional factors
            input_data = sleep_metrics.copy()
            if additional_factors:
                input_data.update(additional_factors)
            
            # Convert to DataFrame
            X = pd.DataFrame([input_data])
            
            # Make base prediction
            if self.base_metadata and 'selected_features' in self.base_metadata:
                required_features = self.base_metadata['selected_features']
                
                # Filter out additional factors that weren't in the training data
                base_X_columns = list(set(required_features).intersection(set(X.columns)))
                
                # Check if we have all the required features
                missing_features = [f for f in required_features if f not in X.columns]
                
                if missing_features:
                    logger.warning(f"Missing features for base model: {missing_features}")
                    # Add missing features with zeros
                    for feature in missing_features:
                        X[feature] = 0
                
                # Remove additional features that weren't in the training data
                extra_features = [f for f in X.columns if f not in required_features]
                if extra_features:
                    logger.warning(f"Removing extra features not used in base model: {extra_features}")
                
                base_X = X[required_features]
            else:
                # Use all common sleep metrics
                sleep_features = [
                    'total_sleep_time', 'wake_time', 'rem_time', 
                    'light_sleep_time', 'deep_sleep_time', 'sleep_efficiency', 
                    'rem_percentage', 'rem_cycles', 'rem_awakenings'
                ]
                # Keep only features that exist in both X and sleep_features
                base_features = [f for f in sleep_features if f in X.columns]
                base_X = X[base_features]
                
                # Add missing features
                missing_features = [f for f in sleep_features if f not in X.columns]
                if missing_features:
                    logger.warning(f"Missing sleep features: {missing_features}")
                    for feature in missing_features:
                        base_X[feature] = 0
            
            base_prediction = bool(self.base_model.predict(base_X)[0])
            base_probability = float(self.base_model.predict_proba(base_X)[0, 1])
            
            # Use calibration model if available
            if self.calibration_model and len(self.personal_data) >= 5:
                # Add base model predictions
                X['base_prediction'] = 1 if base_prediction else 0
                X['base_confidence'] = base_probability
                
                # Get required features for calibration
                calibration_features = self.calibration_model['features']
                missing_features = [f for f in calibration_features if f not in X.columns]
                
                if missing_features:
                    # Add missing features with zeros
                    for feature in missing_features:
                        X[feature] = 0
                
                # Select features
                X_calib = X[calibration_features]
                
                # Make calibrated prediction
                calibration_model = self.calibration_model['model']
                calibrated_prediction = bool(calibration_model.predict(X_calib)[0])
                calibrated_probability = float(calibration_model.predict_proba(X_calib)[0, 1])
                
                # Create result
                result = {
                    'base_prediction': {
                        'good_mood': base_prediction,
                        'probability': base_probability,
                        'mood_score': base_probability * 10
                    },
                    'calibrated_prediction': {
                        'good_mood': calibrated_prediction,
                        'probability': calibrated_probability,
                        'mood_score': calibrated_probability * 10
                    },
                    'has_personalization': True,
                    'confidence': 'high' if len(self.personal_data) >= 10 else 'medium'
                }
            else:
                # Just use the base model prediction
                result = {
                    'good_mood': base_prediction,
                    'probability': base_probability,
                    'mood_score': base_probability * 10,
                    'has_personalization': False,
                    'confidence': 'medium'
                }
            
            logger.info(f"Mood prediction: {result}")
            return result
        
        except Exception as e:
            logger.error(f"Error predicting mood: {e}")
            return None

def simulate_personalization():
    """
    Simulate personalized prediction for a test user.
    """
    # Create a test user
    user_id = "test_user_001"
    predictor = PersonalizedMoodPredictor(user_id)
    
    # Generate sleep data for 10 days
    # Hypothetical patterns: better sleep → better mood
    for i in range(10):
        # Alternate between good and bad sleep
        is_good_sleep = (i % 2 == 0)
        
        if is_good_sleep:
            # Good sleep pattern
            sleep_metrics = {
                'total_sleep_time': np.random.randint(440, 500),  # 7.3-8.3 hours
                'wake_time': np.random.randint(10, 30),
                'rem_time': np.random.randint(100, 130),
                'light_sleep_time': np.random.randint(230, 260),
                'deep_sleep_time': np.random.randint(100, 130),
                'sleep_efficiency': np.random.randint(92, 98),
                'rem_percentage': np.random.randint(22, 28),
                'rem_cycles': 4,
                'rem_awakenings': np.random.randint(0, 3)
            }
            # Good mood (with some randomness)
            mood_rating = min(10, max(0, np.random.normal(8.5, 1.0)))
        else:
            # Poor sleep pattern
            sleep_metrics = {
                'total_sleep_time': np.random.randint(300, 390),  # 5-6.5 hours
                'wake_time': np.random.randint(40, 90),
                'rem_time': np.random.randint(30, 60),
                'light_sleep_time': np.random.randint(180, 230),
                'deep_sleep_time': np.random.randint(50, 90),
                'sleep_efficiency': np.random.randint(70, 85),
                'rem_percentage': np.random.randint(10, 18),
                'rem_cycles': np.random.randint(2, 4),
                'rem_awakenings': np.random.randint(3, 8)
            }
            # Poor mood (with some randomness)
            mood_rating = min(10, max(0, np.random.normal(4.5, 1.5)))
        
        # Add additional factors
        additional_factors = {
            'stress_level': np.random.randint(1, 10),
            'exercise_minutes': np.random.randint(0, 60),
            'caffeine_mg': np.random.randint(0, 300)
        }
        
        # Add record to user data
        predictor.add_sleep_mood_record(
            sleep_metrics, 
            mood_rating, 
            additional_factors['stress_level'],
            additional_factors['exercise_minutes'],
            additional_factors['caffeine_mg']
        )
    
    # Update calibration model
    predictor.update_calibration_model()
    
    # Now test with new sleep data
    new_sleep_metrics = {
        'total_sleep_time': 450,  # 7.5 hours
        'wake_time': 15,
        'rem_time': 110,
        'light_sleep_time': 245,
        'deep_sleep_time': 95,
        'sleep_efficiency': 95,
        'rem_percentage': 24,
        'rem_cycles': 4,
        'rem_awakenings': 1
    }
    
    new_additional_factors = {
        'stress_level': 3,
        'exercise_minutes': 45,
        'caffeine_mg': 150
    }
    
    # Get prediction with personalization
    result = predictor.predict_mood(new_sleep_metrics, new_additional_factors)
    
    if result:
        if result.get('has_personalization', False):
            logger.info("\nPersonalized Prediction:")
            logger.info(f"  Base Model: {'Good' if result['base_prediction']['good_mood'] else 'Bad'} mood, Score: {result['base_prediction']['mood_score']:.1f}/10")
            logger.info(f"  Calibrated: {'Good' if result['calibrated_prediction']['good_mood'] else 'Bad'} mood, Score: {result['calibrated_prediction']['mood_score']:.1f}/10")
            logger.info(f"  Confidence: {result['confidence']}")
        else:
            logger.info("\nBase Model Prediction:")
            logger.info(f"  {'Good' if result['good_mood'] else 'Bad'} mood, Score: {result['mood_score']:.1f}/10")
            logger.info(f"  Confidence: {result['confidence']}")
    else:
        logger.error("Failed to get prediction")

def main():
    """Main function to demonstrate personalized mood prediction."""
    logger.info("Simulating personalized mood prediction")
    simulate_personalization()
    return 0

if __name__ == "__main__":
    sys.exit(main()) 