"""
Medical Triage Model Interface

This module provides the interface for integrating your existing triage classification model.
Replace the MockTriageModel with your actual trained model.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
import joblib
import os

class TriageModel:
    """
    Interface for the triage classification model.
    
    IMPORTANT: Replace this with your actual trained model.
    This is a template that shows the expected interface.
    """
    
    def __init__(self):
        """
        Initialize the triage model.
        
        TODO: Replace this with your actual model loading code.
        Example:
        - self.model = joblib.load('path/to/your/model.pkl')
        - self.scaler = joblib.load('path/to/your/scaler.pkl')
        - self.feature_names = ['age', 'temperature', 'heart_rate', ...]
        """
        self.model =  joblib.load('triage_model.pkl')
        # self.scaler = None  # Your feature scaler goes here
        self.feature_names = self._get_expected_features()
        self.triage_categories = {
            0: {'name': 'Critical', 'color': '#FF4B4B', 'description': 'Immediate attention required'},
            1: {'name': 'Urgent', 'color': '#FF8C00', 'description': 'Treatment within 2 hours'},
            2: {'name': 'Non-urgent', 'color': '#E0E0E0', 'description': 'Treatment when convenient'},
        }

        # color codes: #32CD32, FFD700
        
        # Try to load actual model if it exists
        self.load_model()
    
    def load_model(self):
        """
        Attempt to load your actual model from common paths.
        Add your model loading logic here.
        """
        path = 'triage_model.pkl'
        if os.path.exists(path):
            try:
                self.model = joblib.load(path)
                print(f"✅ Loaded model from {path}")
            except Exception as e:
                print(f"❌ Failed to load model from {path}: {e}")
    
    def _get_expected_features(self) -> List[str]:
        """
        Define the expected feature names for your model.
        
        TODO: Replace this with your actual feature names.
        """
        return [
            'age', 'sex', 'active_bleeding', 'resp_rate', 'heart_rate',
            'systolic_bp', 'diastolic_bp', 'temperature', 'oxygen_sat', 'pregnancy',
            'mode_of_arrival', 'chief_complaint', 'AVPU_scale'
        ]
    
    def _prepare_features(self, patient_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert patient data to model features.
        
        TODO: Implement your actual feature engineering logic here.
        """
        features = []
        
        # Extract numeric features in order
        features.append(patient_data.get('age', 0))
        features.append(1 if patient_data.get('sex', 'Female') == 'Male' else 0)
        features.append(1 if patient_data.get('active_bleeding', 'No') == 'Yes' else 0)
        features.append(patient_data.get('resp_rate', 16))
        features.append(patient_data.get('heart_rate', 80))
        features.append(patient_data.get('systolic_bp', 120))
        features.append(patient_data.get('diastolic_bp', 80))
        features.append(patient_data.get('temperature', 37.0))
        features.append(patient_data.get('oxygen_sat', 98))
        features.append(1 if patient_data.get('pregnancy', 'No') == 'Yes' else 0)

        # Create dummy variables for categorical features
        categorical_data = {
            'mode_of_arrival': patient_data.get('mode_of_arrival', 'Ambulance'),
            'chief_complaint': patient_data.get('chief_complaint', 'Chest Pain'),
            'AVPU_scale': patient_data.get('AVPU_scale', 'Alert')
        }

        # Convert to DataFrame for dummy encoding
        categorical_df = pd.DataFrame([categorical_data])
        dummy_df = pd.get_dummies(categorical_df, drop_first=True)

        # Extend features with dummy variables
        for col in dummy_df.columns:
            features.append(dummy_df[col].iloc[0])

        # Convert to DataFrame with proper column names
        numeric_feature_names = [
            'age', 'sex', 'active_bleeding', 'resp_rate', 'heart_rate',
            'systolic_bp', 'diastolic_bp', 'temperature', 'oxygen_sat', 'pregnancy'
        ]
        categorical_feature_names = dummy_df.columns
        all_feature_names = numeric_feature_names + categorical_feature_names

        return pd.DataFrame([features], columns=all_feature_names)
    
    
    def predict(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make triage prediction for a patient.
        
        Args:
            patient_data: Dictionary containing patient information
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Prepare features
            features = self._prepare_features(patient_data)
            
            if self.model is not None:
                # TODO: Use your actual model here
                # Example:
                # if self.scaler:
                #     features = self.scaler.transform(features)
                prediction = self.model.predict(features)[0]
                probabilities = self.model.predict_proba(features)[0]
                confidence = np.max(probabilities)
                
                # For now, use mock prediction
                return self.load_model(features)
            else:
                # Use mock prediction when no actual model is loaded
                print("⚠️ No trained model found. Using mock prediction for demonstration.")
                
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the model.
        
        TODO: Implement this based on your model type.
        """
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            return importance_dict
        else:
            # Return mock importance for demonstration
            return {feature: np.random.random() for feature in self.feature_names}
    
    # def get_model_info(self) -> Dict[str, Any]:
    #     """
    #     Get information about the loaded model.
    #     """
    #     return {
    #         'model_type': type(self.model).__name__ if self.model else 'Mock Model',
    #         'feature_count': len(self.feature_names),
    #         'categories': list(self.triage_categories.values()),
    #         'is_mock': self.model is None
    #     }
