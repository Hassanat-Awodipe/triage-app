"""
Input form components for the medical triage application.
"""

import streamlit as st
from typing import Dict, Any, List
# from utils import get_vital_signs_ranges


def render_patient_input_form() -> Dict[str, Any]:
    """
    Render the patient information input form.
    
    Returns:
        Dictionary containing all patient data
    """
    patient_data = {}

    # Demographics section
    with st.expander("👤 Patient Demographics", expanded=True):
        patient_data['patent_id'] = st.text_input(
            'Patient ID:', help="Name or ID of the patient")
        col1, col2 = st.columns(2)

        with col1:
            patient_data['age'] = st.number_input(
                "Age (years)",
                min_value=18,
                max_value=80,
                value=None,
                help="Patient's age in years")

        with col2:
            patient_data['sex'] = st.selectbox("Sex",
                                               options=["", "Male", "Female"],
                                               index=0,
                                               help="Patient's gender")

    # Medical status section
    with st.expander("🩸 Medical Status", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            patient_data['active_bleeding'] = st.selectbox(
                "Active Bleeding",
                options=["", "Yes", "No"],
                index=0,
                help="Is the patient actively bleeding?")

        with col2:
            patient_data['pregnancy'] = st.selectbox(
                "Pregnancy",
                options=["", "Yes", "No"],
                index=0,
                help="Is the patient pregnant? (if applicable)")

    # Arrival information section
    with st.expander("🚑 Arrival Information", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            patient_data['mode_of_arrival'] = st.selectbox(
                "Mode of Arrival",
                options=["", "Ambulance", "Walk-in", "Private Vehicle"],
                index=0,
                help="How did the patient arrive at the facility?")

        with col2:
            patient_data['chief_complaint'] = st.selectbox(
                "Chief Complaint",
                options=[
                    "", "Chest pain", "Difficulty breathing", "Headache",
                    "Vomiting or diarrhea", "Fever", "Injury", "Abdominal pain"
                    "Pregnancy-related complication",
                    "Psychiatric/behavioral emergency",
                    "Seizure or loss of consciousness", "Weakness or fatigue",
                    "Other"
                ],
                index=0,
                help="Primary reason for the visit")

    # Vital signs section
    with st.expander("🫀 Vital Signs", expanded=True):

        col1, col2 = st.columns(2)

        with col1:
            patient_data['temperature'] = st.number_input(
                'Temperature (Celsius): ',
                min_value=36.0,
                max_value=39.0,
                value=None,
                step=0.1,
                help="Body temperature in Celsius")

            patient_data['heart_rate'] = st.number_input(
                'Heart Rate: ',
                min_value=60,
                max_value=116,
                value=None,
                help="Heart rate in beats per minute")

            patient_data['resp_rate'] = st.number_input(
                'Respiratory Rate:',
                min_value=12,
                max_value=27,
                value=None,
                help="Breathing rate per minute")

        with col2:
            # Blood pressure inputs
            st.markdown("**Blood Pressure (mmHg)**")
            bp_col1, bp_col2 = st.columns(2)

            with bp_col1:
                patient_data['systolic_bp'] = st.number_input(
                    "Systolic: ",
                    min_value=104,
                    max_value=154,
                    value=None,
                    help="Systolic blood pressure")

            with bp_col2:
                patient_data['diastolic_bp'] = st.number_input(
                    "Diastolic: ",
                    min_value=62,
                    max_value=103,
                    value=None,
                    help="Diastolic blood pressure")

            patient_data['oxygen_sat'] = st.number_input(
                "Oxygen Saturation (%)",
                min_value=81,
                max_value=100,
                value=None,
                help="Blood oxygen saturation percentage")

    # Consciousness assessment section
    with st.expander("🧠 Consciousness Assessment", expanded=True):
        patient_data['AVPU_scale'] = st.selectbox(
            "AVPU Scale",
            options=["", "Alert", "Voice", "Pain", "Unresponsive"],
            index=0,
            help="Patient's level of consciousness using AVPU scale")

    return patient_data


# def render_quick_entry_buttons():
#     """
#     Render quick entry buttons for common scenarios.
#     """
#     st.markdown("### 🚀 Quick Entry Templates")

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         if st.button("😷 Typical Flu Symptoms"):
#             return {
#                 'age': 35,
#                 'sex': 'Female',
#                 'active_bleeding': 'No',
#                 'resp_rate': 18,
#                 'heart_rate': 95,
#                 'systolic_bp': 125,
#                 'diastolic_bp': 80,
#                 'temperature': 38.5,
#                 'oxygen_sat': 97,
#                 'pregnancy': 'No',
#                 'mode_of_arrival': 'Walk-in',
#                 'chief_complaint': 'Fever',
#                 'AVPU_scale': 'Alert',
#                 'Triage_Category': '2'
#             }

#     with col2:
#         if st.button("💔 Chest Pain Scenario"):
#             return {
#                 'age': 58,
#                 'sex': 'Male',
#                 'active_bleeding': 'No',
#                 'resp_rate': 22,
#                 'heart_rate': 110,
#                 'systolic_bp': 150,
#                 'diastolic_bp': 95,
#                 'temperature': 37.2,
#                 'oxygen_sat': 95,
#                 'pregnancy': 'No',
#                 'mode_of_arrival': 'Ambulance',
#                 'chief_complaint': 'Chest Pain',
#                 'AVPU_scale': 'Alert',
#                 'Triage_Category': '0'
#             }

#     with col3:
#         if st.button("🩸 Injury Case"):
#             return {
#                 'age': 28,
#                 'sex': 'Male',
#                 'active_bleeding': 'Yes',
#                 'resp_rate': 24,
#                 'heart_rate': 125,
#                 'systolic_bp': 90,
#                 'diastolic_bp': 60,
#                 'temperature': 36.8,
#                 'oxygen_sat': 92,
#                 'pregnancy': 'No',
#                 'mode_of_arrival': 'Ambulance',
#                 'chief_complaint': 'Injury',
#                 'AVPU_scale': 'Voice',
#                 'Triage_Category': '0'
#             }
