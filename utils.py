"""
Utility functions for the medical triage application.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import io
from datetime import datetime

def validate_inputs(patient_data: Dict[str, Any]) -> List[str]:
    """
    Validate patient input data.
    
    Args:
        patient_data: Dictionary containing patient information
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check required fields
    required_fields = ['age', 'temperature', 'heart_rate', 'blood_pressure_systolic', 
                      'blood_pressure_diastolic', 'respiratory_rate', 'oxygen_saturation']
    
    for field in required_fields:
        if field not in patient_data or patient_data[field] is None:
            errors.append(f"Missing required field: {field}")
    
    # Validate ranges
    if 'age' in patient_data and patient_data['age'] is not None:
        age = patient_data['age']
        if age < 0 or age > 120:
            errors.append("Age must be between 0 and 120 years")
    
    if 'temperature' in patient_data and patient_data['temperature'] is not None:
        temp = patient_data['temperature']
        if temp < 30 or temp > 45:
            errors.append("Temperature must be between 30°C and 45°C")
    
    if 'heart_rate' in patient_data and patient_data['heart_rate'] is not None:
        hr = patient_data['heart_rate']
        if hr < 30 or hr > 250:
            errors.append("Heart rate must be between 30 and 250 bpm")
    
    if 'blood_pressure_systolic' in patient_data and patient_data['blood_pressure_systolic'] is not None:
        sbp = patient_data['blood_pressure_systolic']
        if sbp < 60 or sbp > 250:
            errors.append("Systolic blood pressure must be between 60 and 250 mmHg")
    
    if 'blood_pressure_diastolic' in patient_data and patient_data['blood_pressure_diastolic'] is not None:
        dbp = patient_data['blood_pressure_diastolic']
        if dbp < 30 or dbp > 150:
            errors.append("Diastolic blood pressure must be between 30 and 150 mmHg")
    
    if 'respiratory_rate' in patient_data and patient_data['respiratory_rate'] is not None:
        rr = patient_data['respiratory_rate']
        if rr < 5 or rr > 60:
            errors.append("Respiratory rate must be between 5 and 60 breaths/min")
    
    if 'oxygen_saturation' in patient_data and patient_data['oxygen_saturation'] is not None:
        spo2 = patient_data['oxygen_saturation']
        if spo2 < 50 or spo2 > 100:
            errors.append("Oxygen saturation must be between 50% and 100%")
    
    if 'pain_level' in patient_data and patient_data['pain_level'] is not None:
        pain = patient_data['pain_level']
        if pain < 0 or pain > 10:
            errors.append("Pain level must be between 0 and 10")
    
    # Validate blood pressure relationship
    if ('blood_pressure_systolic' in patient_data and 
        'blood_pressure_diastolic' in patient_data and
        patient_data['blood_pressure_systolic'] is not None and
        patient_data['blood_pressure_diastolic'] is not None):
        
        if patient_data['blood_pressure_systolic'] <= patient_data['blood_pressure_diastolic']:
            errors.append("Systolic blood pressure must be higher than diastolic")
    
    return errors

def format_patient_summary(patient_data: Dict[str, Any], prediction_result: Dict[str, Any]) -> str:
    """
    Generate a formatted patient summary.
    
    Args:
        patient_data: Patient input data
        prediction_result: Model prediction results
        
    Returns:
        Formatted summary string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    summary = f"""
MEDICAL TRIAGE CLASSIFICATION REPORT
Generated: {timestamp}

PATIENT INFORMATION:
- Patient ID: {patient_data.get('patient_id', 'Not provided')}
- Age: {patient_data.get('age', 'Unknown')} years
- Gender: {patient_data.get('gender', 'Not specified')}

VITAL SIGNS:
- Temperature: {patient_data.get('temperature', 'N/A')}°C
- Heart Rate: {patient_data.get('heart_rate', 'N/A')} bpm
- Blood Pressure: {patient_data.get('blood_pressure_systolic', 'N/A')}/{patient_data.get('blood_pressure_diastolic', 'N/A')} mmHg
- Respiratory Rate: {patient_data.get('respiratory_rate', 'N/A')} breaths/min
- Oxygen Saturation: {patient_data.get('oxygen_saturation', 'N/A')}%
- Pain Level: {patient_data.get('pain_level', 'N/A')}/10
- Consciousness: {patient_data.get('consciousness_level', 'N/A')}

PRESENTING SYMPTOMS:
{', '.join(patient_data.get('symptoms', ['None reported']))}

CLINICAL NOTES:
{patient_data.get('clinical_notes', 'No additional notes')}

TRIAGE CLASSIFICATION:
- Level: {prediction_result.get('triage_level', 'Unknown')}
- Category: {prediction_result.get('category', 'Unknown')}
- Confidence: {prediction_result.get('confidence', 0):.1%}
- Recommendation: {prediction_result.get('description', 'No recommendation')}

---
This report was generated by an automated triage classification system.
Clinical judgment should always be used in conjunction with these results.
"""
    return summary

def export_results_to_csv(predictions: List[Dict[str, Any]]) -> str:
    """
    Export prediction results to CSV format.
    
    Args:
        predictions: List of prediction dictionaries
        
    Returns:
        CSV string
    """
    if not predictions:
        return ""
    
    # Prepare data for CSV
    rows = []
    for pred in predictions:
        patient_data = pred.get('patient_data', {})
        row = {
            'Timestamp': pred.get('timestamp', '').strftime('%Y-%m-%d %H:%M:%S') if pred.get('timestamp') else '',
            'Patient_ID': pred.get('patient_id', ''),
            'Age': patient_data.get('age', ''),
            'Gender': patient_data.get('gender', ''),
            'Temperature': patient_data.get('temperature', ''),
            'Heart_Rate': patient_data.get('heart_rate', ''),
            'BP_Systolic': patient_data.get('blood_pressure_systolic', ''),
            'BP_Diastolic': patient_data.get('blood_pressure_diastolic', ''),
            'Respiratory_Rate': patient_data.get('respiratory_rate', ''),
            'Oxygen_Saturation': patient_data.get('oxygen_saturation', ''),
            'Pain_Level': patient_data.get('pain_level', ''),
            'Consciousness_Level': patient_data.get('consciousness_level', ''),
            'Symptoms': ', '.join(patient_data.get('symptoms', [])),
            'Clinical_Notes': patient_data.get('clinical_notes', ''),
            'Triage_Level': pred.get('triage_level', ''),
            'Triage_Category': pred.get('category', ''),
            'Confidence': pred.get('confidence', ''),
            'Description': pred.get('description', '')
        }
        rows.append(row)
    
    # Convert to DataFrame and then CSV
    df = pd.DataFrame(rows)
    
    # Use StringIO to get CSV string
    output = io.StringIO()
    df.to_csv(output, index=False)
    csv_string = output.getvalue()
    output.close()
    
    return csv_string

def get_vital_signs_ranges() -> Dict[str, Dict[str, Any]]:
    """
    Get normal ranges for vital signs for reference.
    
    Returns:
        Dictionary with vital sign ranges
    """
    return {
        'temperature': {
            'normal': (36.0, 37.5),
            'unit': '°C',
            'description': 'Body temperature'
        },
        'heart_rate': {
            'normal': (60, 100),
            'unit': 'bpm',
            'description': 'Heart rate (adults at rest)'
        },
        'blood_pressure_systolic': {
            'normal': (90, 140),
            'unit': 'mmHg',
            'description': 'Systolic blood pressure'
        },
        'blood_pressure_diastolic': {
            'normal': (60, 90),
            'unit': 'mmHg',
            'description': 'Diastolic blood pressure'
        },
        'respiratory_rate': {
            'normal': (12, 20),
            'unit': 'breaths/min',
            'description': 'Respiratory rate (adults)'
        },
        'oxygen_saturation': {
            'normal': (95, 100),
            'unit': '%',
            'description': 'Oxygen saturation'
        }
    }

def calculate_risk_score(patient_data: Dict[str, Any]) -> float:
    """
    Calculate a simple risk score based on patient data.
    
    Args:
        patient_data: Patient information
        
    Returns:
        Risk score between 0 and 1
    """
    score = 0.0
    max_score = 0.0
    
    vital_ranges = get_vital_signs_ranges()
    
    # Check vital signs against normal ranges
    for vital, ranges in vital_ranges.items():
        if vital in patient_data and patient_data[vital] is not None:
            value = patient_data[vital]
            normal_min, normal_max = ranges['normal']
            max_score += 1.0
            
            if value < normal_min:
                # Below normal
                deviation = (normal_min - value) / normal_min
                score += min(deviation, 1.0)
            elif value > normal_max:
                # Above normal
                deviation = (value - normal_max) / normal_max
                score += min(deviation, 1.0)
    
    # Factor in age
    age = patient_data.get('age', 0)
    if age > 65:
        max_score += 1.0
        score += (age - 65) / 35  # Normalize to 0-1 for ages 65-100
    
    # Factor in symptoms
    critical_symptoms = ['chest_pain', 'difficulty_breathing', 'severe_bleeding', 'head_injury']
    symptoms = patient_data.get('symptoms', [])
    
    for symptom in critical_symptoms:
        max_score += 1.0
        if symptom in symptoms:
            score += 1.0
    
    # Factor in pain level
    pain_level = patient_data.get('pain_level', 0)
    if pain_level > 0:
        max_score += 1.0
        score += pain_level / 10.0
    
    # Factor in consciousness level
    consciousness = patient_data.get('consciousness_level', 'Alert')
    consciousness_scores = {
        'Alert': 0.0, 'Confused': 0.3, 'Drowsy': 0.6,
        'Responds to voice': 0.8, 'Unresponsive': 1.0
    }
    max_score += 1.0
    score += consciousness_scores.get(consciousness, 0.0)
    
    # Normalize score
    if max_score > 0:
        return min(score / max_score, 1.0)
    else:
        return 0.0
