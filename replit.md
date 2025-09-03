# Medical Triage Classification System

## Overview

This is a web-based medical triage classification system built with Streamlit that helps healthcare professionals quickly assess patient urgency levels. The application provides a user-friendly interface for inputting patient demographics, vital signs, and symptoms, then uses a machine learning model to classify patients into 5 triage categories (Critical, Urgent, Semi-urgent, Non-urgent, Low priority). The system includes comprehensive input validation, visual result displays with confidence scores, and export capabilities for clinical workflow integration.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Framework**: Single-page web application with reactive components
- **Component-Based Design**: Modular UI components separated into dedicated modules (`input_forms.py`, `result_display.py`)
- **Session State Management**: Persistent storage of predictions and patient data across user interactions
- **Responsive Layout**: Wide layout with expandable sections and column-based organization

### Backend Architecture
- **Model Interface Pattern**: Abstract model interface (`TriageModel`) that can be swapped with different ML implementations
- **Mock Implementation**: Template structure ready for integration with actual trained models
- **Feature Engineering**: Standardized feature extraction and validation pipeline
- **Prediction Pipeline**: End-to-end processing from raw input to classified results

### Data Processing
- **Input Validation**: Comprehensive validation with medical range checking for vital signs
- **Data Transformation**: Structured data handling with pandas DataFrames
- **Export Functionality**: CSV export capabilities for clinical record integration
- **Caching Strategy**: Model caching using Streamlit's `@st.cache_resource` decorator

### Visualization Components
- **Plotly Integration**: Interactive charts for confidence scores and risk visualization
- **Color-Coded Results**: Visual triage level indication with medical color standards
- **Real-time Feedback**: Immediate result display with confidence metrics

## External Dependencies

### Core Framework
- **Streamlit**: Web application framework and UI components
- **Pandas**: Data manipulation and CSV export functionality
- **NumPy**: Numerical computations and array operations

### Visualization
- **Plotly**: Interactive charting and graph generation
- **Plotly Express**: Simplified plotting interface

### Machine Learning (Template)
- **Scikit-learn** (expected): Model training and prediction interface
- **Joblib** (expected): Model serialization and loading capabilities

### Data Export
- **CSV Export**: Built-in pandas functionality for clinical data export
- **IO Operations**: File handling for model loading and data export

Note: The current implementation includes a mock model interface designed to be replaced with actual trained ML models. The system is architected to support various model types including Random Forest, XGBoost, or neural networks through the standardized `TriageModel` interface.