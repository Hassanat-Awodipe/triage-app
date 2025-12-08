"""
Result display components for the medical triage application.
"""

import streamlit as st
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any
from utils import format_patient_summary


def render_prediction_results(prediction_result: Dict[str, Any], explanation_result: Dict[str, Any],
                              patient_data: Dict[str, Any],
                              show_confidence: bool = True):
    """
    Render the triage prediction and explanation results.
    
    Args:
        prediction_result: Model prediction output
        explanation_result: Output explanation
        patient_data: Patient input data
        show_confidence: Whether to show confidence scores
    """

    # Main result display
    triage_level = prediction_result.get('triage_level', 0)
    category = prediction_result.get('category', 'Unknown')
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
        </div>
        """,
        unsafe_allow_html=True
    )

    # Results breakdown
    if show_confidence and 'confidence_scores' in prediction_result:
        col1, col2 = st.columns([1, 1])

        with col1:
            # Confidence scores chart
            confidence_scores = prediction_result['confidence_scores']
            categories = ['Emergency', 'Urgent', 'Non-urgent']
            colors = ['#FF4B4B', '#FFD700', '#32CD32']

            fig = go.Figure(data=[
                go.Bar(
                    x=categories,
                    y=confidence_scores,
                    marker_color=colors,
                    text=[f"{score: .1%}" for score in confidence_scores],
                    textposition='auto',
                )
            ])

            fig.update_layout(
                title="Confidence Scores by Category",
                showlegend=False,
                height=400,
                yaxis=dict(tickformat='.0%')
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Recommendations")

            recommendations = _get_recommendations(triage_level, patient_data)

            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
    else:
        st.markdown("#### Recommendations")

        recommendations = _get_recommendations(triage_level, patient_data)

        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")

    # explain result
    st.markdown("##### Influence of Patient's Indicators on Triage Decision")
    # generate the plot
    explanation_fig, ax = plt.subplots(figsize=(10, 6))
    shap.multioutput_decision_plot(
        explanation_result.get('expected_list', []),
        explanation_result.get('shap_list', []),
        row_index=explanation_result.get('row_index', 0),
        feature_names=explanation_result.get('feature_names', []),
        feature_display_range=slice(-1, 8, -1), # to show the last 7 in descending order,
        highlight=explanation_result.get('highlight', 0),
        legend_labels=explanation_result.get('class_labels_list', []),  # class_labels_list,
        legend_location="lower right",
        show=False
    )
    st.pyplot(explanation_fig, use_container_width=True)

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
            "Establish IV access and monitor vitals continuously"
        ],
        1: [
            "⏰ **URGENT CARE NEEDED**",
            "Patient should be evaluated within 2 hours",
            "Monitor vital signs every 20 minutes",
            "Provide comfort measures as needed"
        ],
        2: [
            "📝 **LOW PRIORITY CARE**",
            "Patient can wait for convenient time slot",
            "Monitor for any deterioration",
            "Patient education regarding condition"
        ]
    }

    recommendations = base_recommendations.get(triage_level, [])

    return recommendations
