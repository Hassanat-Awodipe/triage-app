"""
Input form components for the medical triage application.
"""

import streamlit as st
from typing import Dict, Any, List
from utils import get_vital_signs_ranges

def render_patient_input_form() -> Dict[str, Any]:
    """
    Render the patient information input form.
    
    Returns:
        Dictionary containing all patient data
    """
    patient_data = {}
    
    # Demographics section
    with st.expander("👤 Patient Demographics", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            patient_data['age'] = st.number_input(
                "Age (years)",
                min_value=18,
                max_value=80,
                value=None,
                help="Patient's age in years"
            )
        
        with col2:
            patient_data['sex'] = st.selectbox(
                "Sex",
                options=["", "Male", "Female"],
                index=0,
                help="Patient's gender"
            )
            
    
    # Vital signs section
    with st.expander("🫀 Vital Signs", expanded=True):
        vital_ranges = get_vital_signs_ranges()
        
        col1, col2 = st.columns(2)
        
        with col1:
            temp_range = vital_ranges['temperature']['normal']
            patient_data['temperature'] = st.number_input(
                f"Temperature (°C) - Normal: {temp_range[0]}-{temp_range[1]}°C",
                min_value=30.0,
                max_value=45.0,
                value=None,
                step=0.1,
                help="Body temperature in Celsius"
            )
            
            hr_range = vital_ranges['heart_rate']['normal']
            patient_data['heart_rate'] = st.number_input(
                f"Heart Rate (bpm) - Normal: {hr_range[0]}-{hr_range[1]} bpm",
                min_value=60,
                max_value=116,
                value=None,
                help="Heart rate in beats per minute"
            )
            
            rr_range = vital_ranges['respiratory_rate']['normal']
            patient_data['respiratory_rate'] = st.number_input(
                f"Respiratory Rate (breaths/min) - Normal: {rr_range[0]}-{rr_range[1]} /min",
                min_value=12,
                max_value=27,
                value=None,
                help="Breathing rate per minute"
            )
        
        with col2:
            # Blood pressure inputs
            st.markdown("**Blood Pressure (mmHg)**")
            bp_col1, bp_col2 = st.columns(2)
            
            with bp_col1:
                sbp_range = vital_ranges['blood_pressure_systolic']['normal']
                patient_data['blood_pressure_systolic'] = st.number_input(
                    f"Systolic (Normal: {sbp_range[0]}-{sbp_range[1]})",
                    min_value=60,
                    max_value=250,
                    value=None,
                    help="Systolic blood pressure"
                )
            
            with bp_col2:
                dbp_range = vital_ranges['blood_pressure_diastolic']['normal']
                patient_data['blood_pressure_diastolic'] = st.number_input(
                    f"Diastolic (Normal: {dbp_range[0]}-{dbp_range[1]})",
                    min_value=30,
                    max_value=150,
                    value=None,
                    help="Diastolic blood pressure"
                )
            
            spo2_range = vital_ranges['oxygen_saturation']['normal']
            patient_data['oxygen_saturation'] = st.number_input(
                f"Oxygen Saturation (%) - Normal: {spo2_range[0]}-{spo2_range[1]}%",
                min_value=50,
                max_value=100,
                value=None,
                help="Blood oxygen saturation percentage"
            )
    
    # Pain and consciousness assessment
    with st.expander("🧠 Pain & Consciousness Assessment", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            patient_data['pain_level'] = st.slider(
                "Pain Level (0-10 scale)",
                min_value=0,
                max_value=10,
                value=0,
                help="0 = No pain, 10 = Worst imaginable pain"
            )
        
        with col2:
            patient_data['consciousness_level'] = st.selectbox(
                "Consciousness Level",
                options=["Alert", "Confused", "Drowsy", "Responds to voice", "Unresponsive"],
                index=0,
                help="Patient's level of consciousness"
            )
    
    # Symptoms section
    with st.expander("🩺 Presenting Symptoms", expanded=True):
        st.markdown("**Select all applicable symptoms:**")
        
        # Organize symptoms by category
        symptom_categories = {
            "Cardiovascular": ["chest_pain", "palpitations", "shortness_of_breath"],
            "Respiratory": ["difficulty_breathing", "cough", "wheezing"],
            "Neurological": ["head_injury", "dizziness", "confusion", "seizure"],
            "Gastrointestinal": ["abdominal_pain", "nausea_vomiting", "diarrhea"],
            "General": ["fever", "severe_bleeding", "allergic_reaction", "trauma"]
        }
        
        symptom_labels = {
            "chest_pain": "Chest pain",
            "palpitations": "Palpitations",
            "shortness_of_breath": "Shortness of breath",
            "difficulty_breathing": "Difficulty breathing",
            "cough": "Cough",
            "wheezing": "Wheezing",
            "head_injury": "Head injury",
            "dizziness": "Dizziness",
            "confusion": "Confusion",
            "seizure": "Seizure",
            "abdominal_pain": "Abdominal pain",
            "nausea_vomiting": "Nausea/Vomiting",
            "diarrhea": "Diarrhea",
            "fever": "Fever",
            "severe_bleeding": "Severe bleeding",
            "allergic_reaction": "Allergic reaction",
            "trauma": "Trauma/Injury"
        }
        
        selected_symptoms = []
        
        for category, symptoms in symptom_categories.items():
            st.markdown(f"**{category}:**")
            cols = st.columns(3)
            
            for i, symptom in enumerate(symptoms):
                with cols[i % 3]:
                    if st.checkbox(symptom_labels[symptom], key=f"symptom_{symptom}"):
                        selected_symptoms.append(symptom)
        
        patient_data['symptoms'] = selected_symptoms
    
    # Additional information section
    with st.expander("📝 Additional Information", expanded=False):
        patient_data['clinical_notes'] = st.text_area(
            "Clinical Notes",
            placeholder="Enter any additional clinical observations, history, or notes...",
            height=100,
            help="Additional clinical information that may be relevant"
        )
        
        patient_data['allergies'] = st.text_input(
            "Known Allergies",
            placeholder="List any known allergies",
            help="Patient's known allergies or sensitivities"
        )
        
        patient_data['medications'] = st.text_area(
            "Current Medications",
            placeholder="List current medications...",
            height=60,
            help="Current medications patient is taking"
        )
        
        patient_data['medical_history'] = st.text_area(
            "Relevant Medical History",
            placeholder="Enter relevant medical history...",
            height=60,
            help="Relevant past medical history"
        )
    
    return patient_data

def render_quick_entry_buttons():
    """
    Render quick entry buttons for common scenarios.
    """
    st.markdown("### 🚀 Quick Entry Templates")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("😷 Typical Flu Symptoms"):
            return {
                'temperature': 38.5,
                'heart_rate': 95,
                'blood_pressure_systolic': 125,
                'blood_pressure_diastolic': 80,
                'respiratory_rate': 18,
                'oxygen_saturation': 97,
                'pain_level': 3,
                'consciousness_level': 'Alert',
                'symptoms': ['fever', 'cough', 'nausea_vomiting']
            }
    
    with col2:
        if st.button("💔 Chest Pain Scenario"):
            return {
                'temperature': 37.2,
                'heart_rate': 110,
                'blood_pressure_systolic': 150,
                'blood_pressure_diastolic': 95,
                'respiratory_rate': 22,
                'oxygen_saturation': 95,
                'pain_level': 8,
                'consciousness_level': 'Alert',
                'symptoms': ['chest_pain', 'shortness_of_breath', 'dizziness']
            }
    
    with col3:
        if st.button("🩸 Trauma Case"):
            return {
                'temperature': 36.8,
                'heart_rate': 125,
                'blood_pressure_systolic': 90,
                'blood_pressure_diastolic': 60,
                'respiratory_rate': 24,
                'oxygen_saturation': 92,
                'pain_level': 9,
                'consciousness_level': 'Confused',
                'symptoms': ['severe_bleeding', 'trauma', 'head_injury']
            }
    
    return None
