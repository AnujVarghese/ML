"""
Home/Landing Page View Module (Presentation Layer)
--------------------------------------------------
This module defines the introductory and tutorial view for the AutoML Suite.

Design Rationale:
- Component-driven architecture: This function is completely self-contained and 
  exposed as a callable method. It is dynamically executed by the Streamlit App Router (main.py).
- Aesthetic cards: Employs HTML containers styled with Streamlit-compatible CSS to create
  sleek cards detailing the functionality of each workspace.
"""

import streamlit as st

def show_home():
    """
    Renders the Landing Page layout on Streamlit.
    """
    # Header Hero Card
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 3.5rem 0;">
        <h1 style="font-size: 3.2rem; font-weight: 800; background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 50%, #EC4899 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -1px; margin-bottom: 0.75rem;">
            Enterprise AutoML Benchmarking Dashboard
        </h1>
        <p style="font-size: 1.25rem; max-width: 820px; margin: 0 auto; opacity: 0.85; line-height: 1.6; font-weight: 300;">
            A modular benchmarking application that automates data cleaning, scaling, training, 
            and comparative model diagnostics. Upload a dataset or load fallback schemas 
            to start experimenting with ML pipelines.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Workspace Cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color: var(--secondary-background-color); border: 1px solid rgba(128,128,128,0.15); border-radius: 12px; padding: 2rem; height: 100%; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
            <div style="font-size: 2.75rem; margin-bottom: 0.75rem;">📈</div>
            <h3 style="margin-top: 0; font-size: 1.4rem; font-weight: 700;">Regression Workspace</h3>
            <p style="opacity: 0.8; font-size: 0.95rem; line-height: 1.5; margin-bottom: 1.25rem;">
                Train and compare 6 distinct algorithms on continuous target distributions. 
                Evaluate error variance and correlation fits.
            </p>
            <strong style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.7; display: block; margin-bottom: 0.5rem;">Supported Algorithms:</strong>
            <ul style="opacity: 0.8; font-size: 0.95rem; line-height: 1.6; margin-left: 1.2rem; padding-left: 0;">
                <li>Linear Regression & Ridge Regression</li>
                <li>Random Forest Regressor</li>
                <li>Gradient Boosting Regressor</li>
                <li>Support Vector Regressor (SVR)</li>
                <li>K-Neighbors Regressor</li>
            </ul>
            <p style="opacity: 0.8; font-size: 0.95rem; margin-top: 1rem; border-top: 1px solid rgba(128,128,128,0.1); padding-top: 0.75rem;">
                📊 <strong>Key Metrics:</strong> R², MAE, MSE, RMSE
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div style="background-color: var(--secondary-background-color); border: 1px solid rgba(128,128,128,0.15); border-radius: 12px; padding: 2rem; height: 100%; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
            <div style="font-size: 2.75rem; margin-bottom: 0.75rem;">🎯</div>
            <h3 style="margin-top: 0; font-size: 1.4rem; font-weight: 700;">Classification Workspace</h3>
            <p style="opacity: 0.8; font-size: 0.95rem; line-height: 1.5; margin-bottom: 1.25rem;">
                Evaluate class boundaries and prediction confidence scores. Compares accuracy, precision, 
                recall, and macro-averaged scores.
            </p>
            <strong style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.7; display: block; margin-bottom: 0.5rem;">Supported Algorithms:</strong>
            <ul style="opacity: 0.8; font-size: 0.95rem; line-height: 1.6; margin-left: 1.2rem; padding-left: 0;">
                <li>Logistic Regression</li>
                <li>Random Forest Classifier</li>
                <li>Gradient Boosting Classifier</li>
                <li>Support Vector Classifier (SVC)</li>
                <li>K-Nearest Neighbors (KNN)</li>
                <li>Decision Tree Classifier</li>
            </ul>
            <p style="opacity: 0.8; font-size: 0.95rem; margin-top: 1rem; border-top: 1px solid rgba(128,128,128,0.1); padding-top: 0.75rem;">
                📊 <strong>Key Metrics:</strong> Accuracy, Precision, Recall, F1-Score
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("### 🏛️ Engineering & Modular Architecture Details")
    st.markdown("""
    This project is refactored into a highly modular, decoupled enterprise application structure:
    
    * **Backend (Pure Python Engines)**:
        * `core/data_handler.py` ➔ Safely parses CSV/Excel buffers and caches fallback data.
        * `core/preprocessor.py` ➔ Runs robust, leakage-free cleanup, imputation, scaling, and train-test splits.
        * `core/ml_engine.py` ➔ Executes dictionary loops over model classifiers/regressors and computes metrics.
    * **Frontend (Streamlit Presentation Layer)**:
        * `ui/home.py` ➔ Landing page routing view (this page).
        * `ui/regression_ui.py` ➔ Layout, interactive tables with dynamic CSS winner-highlighting, and comparative Plotly distributions.
        * `ui/classification_ui.py` ➔ Layout, metrics dashboard, multi-class confidence probability charts.
        * `main.py` ➔ Routing controller using `st.navigation` page endpoints.
    
    *Explore the sidebar navigation to get started with a workspace!*
    """)
