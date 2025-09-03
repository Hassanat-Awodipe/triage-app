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
               'systolic_bp', 'diastolic_bp', 'temperature', 'oxygen_sat', 'pregnancy', 'mode_of_arrival', 'chief_complaint', 'AVPU_scale', 'Triage_Category'
        ]
    
    def _prepare_features(self, patient_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert patient data to model features.
        
        TODO: Implement your actual feature engineering logic here.
        """
        features = []
        
        # Extract numeric features
        features.append(patient_data.get('age', 0))
        features.append(patient_data.get('resp_rate', 16))
        features.append(patient_data.get('heart_rate', 80))
        features.append(patient_data.get('systolic_bp', 120))
        features.append(patient_data.get('diastolic_bp', 80))
        features.append(patient_data.get('temperature', 37.0))
        features.append(patient_data.get('oxygen_sat', 98))


        # Extract categorical features

        # Convert binary features to numeric 
        features.append(1 if patient_data.get('sex', 'Female') else 0)
        features.append(1 if patient_data.get('active_bleeding', 'Yes') else 0)
        features.append(1 if patient_data.get('pregnancy', 'Yes') else 0)

        # create dummy variables for categorical features
        features.append(patient_data.get('mode_of_arrival', 'Ambulance'))
        features.append(patient_data.get('AVPU_scale', 'Alert'))
        features.append(patient_data.get('chief_complaint', 'Chest Pain'))

        # Create dummy variables for categorical features
        categorical_data = {
            'mode_of_arrival': patient_data.get('mode_of_arrival', 'Ambulance'),
            'AVPU_scale': patient_data.get('AVPU_scale', 'Alert'),
            'chief_complaint': patient_data.get('chief_complaint', 'Chest Pain')
        }

        # Convert to DataFrame for dummy encoding
        categorical_df = pd.DataFrame([categorical_data])
        dummy_df = pd.get_dummies(categorical_df, drop_first=True)

        # Extend features with dummy variables
        for col in dummy_df.columns:
            features.append(dummy_df[col].iloc[0])

        # Convert to DataFrame with proper column names
        numeric_feature_names = [
            'age', 'resp_rate', 'heart_rate', 'systolic_bp', 'diastolic_bp', 
            'temperature', 'oxygen_sat', 'sex', 'active_bleeding', 'pregnancy'
        ]
        all_feature_names = numeric_feature_names + dummy_df.columns

        return pd.DataFrame([features], columns=all_feature_names)

        # return pd.DataFrame(features)
    
    # def _mock_prediction(self, features: np.ndarray) -> Dict[str, Any]:
    #     """
    #     Mock prediction for demonstration purposes.
        
    #     TODO: Remove this when you integrate your actual model.
    #     """
    #     # Simple rule-based mock logic for demonstration
    #     age = features[0, 0]
    #     temperature = features[0, 1]
    #     heart_rate = features[0, 2]
    #     bp_systolic = features[0, 3]
    #     oxygen_sat = features[0, 6]
    #     pain_level = features[0, 7]
    #     consciousness = features[0, 8]
        
    #     # Count critical symptoms (last 9 features are symptoms)
    #     critical_symptoms = np.sum(features[0, 9:])
        
    #     # Mock triage logic
    #     if (consciousness <= 1 or oxygen_sat < 90 or bp_systolic > 180 or 
    #         temperature > 40 or critical_symptoms >= 3):
    #         triage_level = 1
    #         confidence = 0.95
    #     elif (pain_level >= 8 or heart_rate > 120 or temperature > 38.5 or 
    #           critical_symptoms >= 2):
    #         triage_level = 2
    #         confidence = 0.85
    #     elif (pain_level >= 5 or heart_rate > 100 or temperature > 38 or 
    #           critical_symptoms >= 1):
    #         triage_level = 3
    #         confidence = 0.75
    #     elif pain_level >= 3 or age > 65:
    #         triage_level = 4
    #         confidence = 0.70
    #     else:
    #         triage_level = 5
    #         confidence = 0.80
        
    #     # Generate mock confidence scores for all categories
    #     confidence_scores = np.random.dirichlet([1] * 5)
    #     confidence_scores[triage_level - 1] = confidence
    #     confidence_scores = confidence_scores / confidence_scores.sum()
        
    #     return {
    #         'triage_level': int(triage_level),
    #         'confidence': float(confidence),
    #         'confidence_scores': confidence_scores.tolist(),
    #         'category': self.triage_categories[triage_level]['name'],
    #         'color': self.triage_categories[triage_level]['color'],
    #         'description': self.triage_categories[triage_level]['description']
    #     }
    
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
