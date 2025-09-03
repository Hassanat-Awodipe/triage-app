"""
Result display components for the medical triage application.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any
from utils import format_patient_summary, calculate_risk_score

def render_prediction_results(prediction_result: Dict[str, Any], patient_data: Dict[str, Any], show_confidence: bool = True):
    """
    Render the triage prediction results.
    
    Args:
        prediction_result: Model prediction output
        patient_data: Patient input data
        show_confidence: Whether to show confidence scores
    """
    
    # Main result display
    triage_level = prediction_result.get('triage_level', 0)
    category = prediction_result.get('category', 'Unknown')
    confidence = prediction_result.get('confidence', 0)
    color = prediction_result.get('color', '#gray')
    description = prediction_result.get('description', 'No description available')
    
    # Large result card
    st.markdown(
        f"""
        <div style="
            padding: 20px;
            border-radius: 10px;
            border: 3px solid {color};
            background-color: {color}20;
            text-align: center;
            margin: 20px 0;
        ">
            <h1 style="color: {color}; margin: 0; font-size: 2.5em;">
                Level {triage_level} - {category}
            </h1>
            <p style="font-size: 1.2em; margin: 10px 0; color: #333;">
                {description}
            </p>
            <p style="font-size: 1.1em; margin: 5px 0; color: #666;">
                Confidence: {confidence:.1%}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Results breakdown
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Classification Details")
        
        # Confidence scores chart
        if show_confidence and 'confidence_scores' in prediction_result:
            confidence_scores = prediction_result['confidence_scores']
            categories = ['Critical', 'Urgent', 'Semi-urgent', 'Non-urgent', 'Low priority']
            colors = ['#FF4B4B', '#FF8C00', '#FFD700', '#32CD32', '#E0E0E0']
            
            fig = go.Figure(data=[
                go.Bar(
                    x=categories,
                    y=confidence_scores,
                    marker_color=colors,
                    text=[f"{score:.1%}" for score in confidence_scores],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Confidence Scores by Category",
                xaxis_title="Triage Category",
                yaxis_title="Confidence Score",
                showlegend=False,
                height=400,
                yaxis=dict(tickformat='.0%')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk factors
        st.subheader("⚠️ Risk Assessment")
        
        risk_score = calculate_risk_score(patient_data)
        risk_color = '#FF4B4B' if risk_score > 0.7 else '#FF8C00' if risk_score > 0.4 else '#32CD32'
        
        st.markdown(
            f"""
            <div style="
                padding: 15px;
                border-radius: 8px;
                background-color: {risk_color}20;
                border-left: 5px solid {risk_color};
                margin: 10px 0;
            ">
                <h4 style="margin: 0; color: {risk_color};">
                    Overall Risk Score: {risk_score:.1%}
                </h4>
                <p style="margin: 5px 0; color: #666;">
                    {'High risk - requires immediate attention' if risk_score > 0.7 else 
                     'Moderate risk - monitor closely' if risk_score > 0.4 else 
                     'Low risk - routine care appropriate'}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.subheader("📋 Clinical Summary")
        
        # Vital signs status
        vital_status = _get_vital_signs_status(patient_data)
        
        for vital, status in vital_status.items():
            icon = "🔴" if status['status'] == 'critical' else "🟡" if status['status'] == 'abnormal' else "🟢"
            st.markdown(f"{icon} **{status['name']}**: {status['value']} {status['unit']} {status['note']}")
        
        # Symptoms summary
        symptoms = patient_data.get('symptoms', [])
        if symptoms:
            st.markdown("**🩺 Presenting Symptoms:**")
            for symptom in symptoms:
                symptom_formatted = symptom.replace('_', ' ').title()
                severity_icon = "🔴" if symptom in ['chest_pain', 'difficulty_breathing', 'severe_bleeding', 'head_injury'] else "🟡"
                st.markdown(f"{severity_icon} {symptom_formatted}")
        else:
            st.markdown("**🩺 Presenting Symptoms:** None reported")
        
        # Pain and consciousness
        pain_level = patient_data.get('pain_level', 0)
        consciousness = patient_data.get('consciousness_level', 'Alert')
        
        pain_icon = "🔴" if pain_level >= 8 else "🟡" if pain_level >= 5 else "🟢"
        consciousness_icon = "🔴" if consciousness in ['Unresponsive', 'Responds to voice'] else "🟡" if consciousness in ['Drowsy', 'Confused'] else "🟢"
        
        st.markdown(f"{pain_icon} **Pain Level:** {pain_level}/10")
        st.markdown(f"{consciousness_icon} **Consciousness:** {consciousness}")
    
    # Action recommendations
    st.subheader("🎯 Recommended Actions")
    
    recommendations = _get_recommendations(triage_level, patient_data)
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")
    
    # Patient summary download
    st.subheader("📄 Patient Summary")
    
    summary = format_patient_summary(patient_data, prediction_result)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.text_area(
            "Generated Summary",
            value=summary,
            height=200,
            help="Complete patient summary with triage results"
        )
    
    with col2:
        st.download_button(
            label="📥 Download Summary",
            data=summary,
            file_name=f"triage_summary_{patient_data.get('patient_id', 'unknown')}.txt",
            mime="text/plain",
            use_container_width=True
        )

def _get_vital_signs_status(patient_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate vital signs status.
    
    Args:
        patient_data: Patient data dictionary
        
    Returns:
        Dictionary with vital signs status
    """
    vital_ranges = {
        'temperature': {'normal': (36.0, 37.5), 'critical': (35.0, 40.0), 'unit': '°C', 'name': 'Temperature'},
        'heart_rate': {'normal': (60, 100), 'critical': (40, 150), 'unit': 'bpm', 'name': 'Heart Rate'},
        'blood_pressure_systolic': {'normal': (90, 140), 'critical': (70, 180), 'unit': 'mmHg', 'name': 'Systolic BP'},
        'blood_pressure_diastolic': {'normal': (60, 90), 'critical': (40, 110), 'unit': 'mmHg', 'name': 'Diastolic BP'},
        'respiratory_rate': {'normal': (12, 20), 'critical': (8, 30), 'unit': '/min', 'name': 'Respiratory Rate'},
        'oxygen_saturation': {'normal': (95, 100), 'critical': (85, 100), 'unit': '%', 'name': 'Oxygen Saturation'}
    }
    
    status = {}
    
    for vital, ranges in vital_ranges.items():
        if vital in patient_data and patient_data[vital] is not None:
            value = patient_data[vital]
            normal_min, normal_max = ranges['normal']
            critical_min, critical_max = ranges['critical']
            
            if value < critical_min or value > critical_max:
                vital_status = 'critical'
                note = '(Critical)'
            elif value < normal_min or value > normal_max:
                vital_status = 'abnormal'
                note = '(Abnormal)'
            else:
                vital_status = 'normal'
                note = '(Normal)'
            
            status[vital] = {
                'status': vital_status,
                'value': value,
                'unit': ranges['unit'],
                'name': ranges['name'],
                'note': note
            }
    
    return status

def _get_recommendations(triage_level: int, patient_data: Dict[str, Any]) -> list:
    """
    Generate action recommendations based on triage level and patient data.
    
    Args:
        triage_level: Triage classification level (1-5)
        patient_data: Patient data dictionary
        
    Returns:
        List of recommendation strings
    """
    base_recommendations = {
        0: [
            "🚨 **IMMEDIATE INTERVENTION REQUIRED**",
            "Prepare for immediate medical assessment",
            "Consider advanced airway management if needed",
            "Establish IV access and monitor vitals continuously",
            "Prepare for potential emergency procedures"
        ],
        1: [
            "⏰ **URGENT CARE NEEDED**",
            "Patient should be evaluated within 2 hours",
            "Monitor vital signs every 30 minutes",
            "Provide comfort measures as needed",
            "Reassess if condition changes"
        ],
        2: [
            "📝 **LOW PRIORITY CARE**",
            "Patient can wait for convenient time slot",
            "Monitor for any deterioration",
            "Basic comfort measures sufficient",
            "Patient education regarding condition",
            
        ]
    }
    
    recommendations = base_recommendations.get(triage_level, [])
    
    return recommendations

# def render_trend_analysis(predictions_history: list):
#     """
#     Render trend analysis for multiple predictions.
    
#     Args:
#         predictions_history: List of historical predictions
#     """
#     if len(predictions_history) < 2:
#         return
    
#     st.subheader("📈 Trend Analysis")
    
#     # Create trend data
#     df = pd.DataFrame([
#         {
#             'timestamp': pred['timestamp'],
#             'triage_level': pred['triage_level'],
#             'confidence': pred['confidence'],
#             'patient_id': pred.get('patient_id', 'Unknown')
#         }
#         for pred in predictions_history
#     ])
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         # Triage level over time
#         fig = px.line(
#             df, 
#             x='timestamp', 
#             y='triage_level',
#             title='Triage Level Trend',
#             markers=True
#         )
#         fig.update_yaxis(dtick=1)
#         st.plotly_chart(fig, use_container_width=True)
    
#     with col2:
#         # Confidence over time
#         fig = px.line(
#             df, 
#             x='timestamp', 
#             y='confidence',
#             title='Prediction Confidence Trend',
#             markers=True
#         )
#         fig.update_yaxis(tickformat='.0%')
#         st.plotly_chart(fig, use_container_width=True)
