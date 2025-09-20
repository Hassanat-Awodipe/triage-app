import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model_interface import TriageModel
from components.input_forms import render_patient_input_form
from components.result_display import render_prediction_results
from utils import validate_inputs, export_results_to_csv

# Page configuration
st.set_page_config(
    page_title="Medical Triage Classification System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
UNDER_CONSTRUCTION = True  # 🔧 change this later

if UNDER_CONSTRUCTION:
    st.title("AI-Assisted Medical Triage for Resource-Limited Settings - Nigeria")
    st.write("We’re updating the app 🚧. Please check back soon!")
# else:
# # Initialize session state
# if 'predictions' not in st.session_state:
#     st.session_state.predictions = []
# if 'patient_data' not in st.session_state:
#     st.session_state.patient_data = {}
#
#
# # Initialize the triage model
# @st.cache_resource
# def load_triage_model():
#     """Load and cache the triage classification model"""
#     return TriageModel()
#
#
# def main():
#     # Header
#     st.title("AI-Assisted Medical Triage for Resource-Limited Settings - Nigeria")
#     st.markdown("---")
#
#     # Sidebar for instructions and settings
#     with st.sidebar:
#         st.markdown("""
#         **How to use this system:**
#         1. Fill in patient information on the left panel
#         2. Enter vital signs measurements
#         3. Select chief complaint. Only one complaint can be selected as the chief complaint. This is usually what brings the patient to the facility.
#         4. Click 'Classify Triage Level' for results
#
#         **Triage Categories:**
#         - 🔴 **Emergency** (0): Immediate attention required
#         - 🟡 **Urgent** (1): Treatment within 2 hours
#         - 🟢 **Non-urgent** (2): Treatment within 4 hours
#         """)
#
#         st.markdown("---")
#
#         # Settings
#         show_confidence = st.checkbox("Show confidence scores", value=True)
#         auto_predict = st.checkbox("Auto-predict on input change", value=False)
#
#         st.markdown("---")
#
#         # Export options
#         if st.session_state.predictions:
#             if st.button("📥 Download All Results"):
#                 csv_data = export_results_to_csv(st.session_state.predictions)
#                 st.download_button(
#                     label="📄 Download CSV",
#                     data=csv_data,
#                     file_name=f"triage_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                     mime="text/csv"
#                 )
#
#         st.markdown("---")
#         st.subheader('Model Information')
#         model = load_triage_model()
#         info = model.get_model_info()
#
#         st.write(f"**Model Type: ** {info['model_type']}")
#         st.write(f"**Feature Count: ** {info['feature_count']}")
#         # st.write(f"**Number of training data:** {info['train_data']}")
#
#     # Main content area
#     col1, col2 = st.columns([1, 1])
#
#     with col1:
#         st.header("📝 Patient Information")
#
#         # Render patient input form
#         patient_data = render_patient_input_form()
#
#         # Store in session state
#         st.session_state.patient_data = patient_data
#
#         # Prediction button
#         predict_button = st.button(
#             "Get Triage",
#             type="primary",
#             width='stretch'
#         )
#
#         # Auto-predict if enabled and data is valid
#         if auto_predict and validate_inputs(patient_data):
#             predict_button = True
#
#     with col2:
#         st.header("📊 Triage Classification Results")
#
#         if predict_button or (auto_predict and validate_inputs(patient_data)):
#             # Validate inputs
#             validation_errors = validate_inputs(patient_data)
#
#             if validation_errors:
#                 st.error("Please correct the following errors:")
#                 for error in validation_errors:
#                     st.error(f"• {error}")
#             else:
#                 # Show progress bar
#                 progress_bar = st.progress(0)
#                 status_text = st.empty()
#
#                 try:
#                     # Load model and make prediction
#                     status_text.text("Loading triage model...")
#                     progress_bar.progress(25)
#
#                     # model = load_triage_model()
#
#                     status_text.text("Processing patient data...")
#                     progress_bar.progress(50)
#
#                     # Make prediction
#                     prediction_result = model.predict(patient_data)
#
#                     status_text.text("Generating results...")
#                     progress_bar.progress(75)
#
#                     # Store prediction in session state
#                     prediction_with_timestamp = {
#                         **prediction_result,
#                         'timestamp': datetime.now(),
#                         'patient_id': patient_data.get('patient_id', 'Unknown'),
#                         'patient_data': patient_data.copy()
#                     }
#                     st.session_state.predictions.append(prediction_with_timestamp)
#
#                     progress_bar.progress(100)
#                     status_text.text("Classification complete!")
#
#                     # Clear progress indicators
#                     progress_bar.empty()
#                     status_text.empty()
#
#                     # Display results
#                     render_prediction_results(
#                         prediction_result,
#                         patient_data,
#                         show_confidence=show_confidence
#                     )
#
#                 except Exception as e:
#                     progress_bar.empty()
#                     status_text.empty()
#                     st.error(f"Error during prediction: {str(e)}")
#                     st.info("Please check your model configuration and try again.")
#
#         else:
#             # Show placeholder when no prediction is made
#             st.info("👆 Complete the patient information form and click 'Classify Triage Level' to see results.")
#
#     # Footer with recent predictions
#     if st.session_state.predictions:
#         st.markdown("---")
#         st.header("📈 Recent Classifications")
#
#         # Show last 5 predictions in a table
#         recent_predictions = st.session_state.predictions[-5:]
#
#         display_data = []
#         for pred in recent_predictions:
#             display_data.append({
#                 'Time': pred['timestamp'].strftime('%H:%M:%S'),
#                 'Patient ID': pred['patient_id'],
#                 'Triage Level': pred['triage_level'],
#                 'Category': pred['category'],
#                 'Confidence': f"{pred['confidence']:.1%}"
#             })
#
#         df = pd.DataFrame(display_data)
#         st.dataframe(df, width='stretch')
#
#
# if __name__ == "__main__":
#     main()
