"""
Medical Triage Model Interface

This module provides the interface for integrating the existing triage classification model.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import shap
import plotly.express as px
import joblib
import os


class TriageModel:
    """
    Interface for the triage classification model.
    """

    def __init__(self):
        """
        Initialize the triage model.

        """
        self.model = joblib.load('best_triage_model.pkl')
        self.feature_names = self._get_expected_features()
        self.triage_categories = {
            0: {'name': 'Emergency', 'color': '#FF4B4B', 'description': 'Immediate attention required'},
            1: {'name': 'Urgent', 'color': '#FFD700', 'description': 'Treatment within 2 hours'},
            2: {'name': 'Non-urgent', 'color': '#32CD32', 'description': 'Treatment within 3 hours'},
        }

        # load model
        self.load_model()

    def load_model(self):
        """
        Attempt to load the model from path.

        """
        path = 'best_triage_model.pkl'
        if os.path.exists(path):
            try:
                self.model = joblib.load(path)
                print(f"✅ Loaded model from {path}")
            except Exception as e:
                print(f"❌ Failed to load model from {path}: {e}")

    def _get_expected_features(self) -> List[str]:
        """
        Define the expected feature names for the model.

        """
        return [
            'age', 'sex', 'active_bleeding', 'resp_rate', 'heart_rate',
            'systolic_bp', 'diastolic_bp', 'temperature', 'oxygen_sat', 'pregnancy',
            'mode_of_arrival', 'chief_complaint', 'AVPU_scale'
        ]

    def _prepare_features(self, patient_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert patient data to model features compatible with triage_model (2).pkl.
        
        Returns a DataFrame with numeric features and dummy-encoded categorical features.
        """
        # Prepare numeric features in the exact order expected by the model
        numeric_features = {
            'age': patient_data.get('age', 0),
            'sex': 1 if patient_data.get('sex', 'Male') == 'Female' else 0,
            'active_bleeding': 1 if patient_data.get('active_bleeding', 'No') == 'Yes' else 0,
            'resp_rate': patient_data.get('resp_rate', 16),
            'heart_rate': patient_data.get('heart_rate', 80),
            'systolic_bp': patient_data.get('systolic_bp', 120),
            'diastolic_bp': patient_data.get('diastolic_bp', 80),
            'temperature': patient_data.get('temperature', 37.0),
            'oxygen_sat': patient_data.get('oxygen_sat', 98),
            'pregnancy': 1 if patient_data.get('pregnancy', 'No') == 'Yes' else 0
        }

        # Create numeric DataFrame
        numeric_df = pd.DataFrame([numeric_features])

        # Prepare categorical features for dummy encoding
        categorical_data = {
            'mode_of_arrival': patient_data.get('mode_of_arrival', 'Walk-in'),
            'chief_complaint': patient_data.get('chief_complaint', 'Abdominal pain'),
            'AVPU_scale': patient_data.get('AVPU_scale', 'Alert')
        }

        # Create DataFrame for dummy encoding
        categorical_df = pd.DataFrame([categorical_data])

        # Create dummy variables for all categorical features
        dummy_df = pd.get_dummies(categorical_df).astype(int)

        # Define expected dummy columns based on the training data
        expected_dummy_columns = [
            'mode_of_arrival_Private vehicle', 'mode_of_arrival_Walk-in',
            'chief_complaint_Abdominal pain', 'chief_complaint_Chest pain',
            'chief_complaint_Difficulty breathing', 'chief_complaint_Fever',
            'chief_complaint_Headache', 'chief_complaint_Injury',
            'chief_complaint_Pregnancy-related complication',
            'chief_complaint_Psychiatric/behavioral emergency',
            'chief_complaint_Seizure or loss of consciousness',
            'chief_complaint_Vomiting or diarrhea',
            'chief_complaint_Weakness or fatigue', 'AVPU_scale_Pain',
            'AVPU_scale_Unresponsive', 'AVPU_scale_Voice'
        ]

        # Reindex to ensure all expected columns are present with 0 for missing ones
        dummy_df = dummy_df.reindex(columns=expected_dummy_columns, fill_value=0)

        # Combine numeric and categorical features
        result_df = pd.concat([numeric_df, dummy_df], axis=1)

        return result_df

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
                prediction = self.model.predict(features)[0]
                probabilities = self.model.predict_proba(features)[0]
                confidence = np.max(probabilities)

                # Get category info
                category_info = self.triage_categories.get(prediction, {
                    'name': 'Unknown',
                    'color': '#808080',
                    'description': 'Unknown category'
                })

                return {
                    'triage_level': int(prediction),
                    'category': category_info['name'],
                    'confidence': float(confidence),
                    'confidence_scores': probabilities.tolist(),
                    'description': category_info['description'],
                    'color': category_info['color']
                }
            else:
                raise Exception("No trained model loaded. Please check triage_model.pkl file.")

        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")

    # calculate feature importance
    def get_feature_importance(self):
        """
        Get feature importance scores from the model.
        """
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            importance_percent = 100 * (importance / importance.sum())
            feat_importance = pd.DataFrame({
                'Feature': self.model.feature_names_in_,
                'Importance (%)': importance_percent
            }).sort_values('Importance (%)', ascending=False)
            # Create a fig to show the importance
            importance_fig = px.bar(feat_importance,
                                    x='Importance (%)',
                                    y='Feature',
                                    orientation='h'
                                    )
            importance_fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},  # highest on top
                margin=dict(l=150, r=40, t=60, b=40),
                height=400 + 20 * len(feat_importance)
            )
            importance_fig.update_yaxes(tickfont=dict(size=10))
            return feat_importance, importance_fig
        else:
            raise AttributeError("The model does not have feature_importances_")

    # calculate shap values
    def explain(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get feature contribution for predicted category
        """
        try:
            features_df = self._prepare_features(patient_data)
            row_index = 0
            probabilities = self.model.predict_proba(features_df)

            # SHAP explainer
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(features_df)

            # labels for the legend
            class_labels_list = [
                f"{['Emergency', 'Urgent', 'Non-urgent'][i]} ({probabilities[row_index, i].round(2):.2f})" for i
                in self.model.classes_]

            return {
                'expected_list': explainer.expected_value.tolist(),
                'shap_list': [shap_values[..., i] for i in range(shap_values.shape[-1])],
                'row_index': row_index,
                'feature_names': self.model.feature_names_in_,
                'class_labels_list': class_labels_list,
                'highlight': [np.argmax(probabilities[row_index])]
            }

        except Exception as e:
            raise Exception(f"Cannot Calculate Shap Value: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        """

        return {
            'model_type': type(self.model).__name__ if self.model else 'No Model Found',
            'feature_count': len(self.feature_names),
        }
