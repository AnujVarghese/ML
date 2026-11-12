"""
Regression Workspace View Module (Presentation Layer)
-----------------------------------------------------
Handles UI widgets, Plotly visualizations, prediction inputs, and layout.
Links imports from the backend engines (in core/) to feed live analytics.

Design Rationale:
- Decoupling of Concerns: UI code remains here. ML logic remains in core/.
- Session State Caching: Results are cached based on configuration variables (dataset source,
  target, random seed, training ratio, row limit). Prediction playground interactions
  retrieve cached objects instead of triggering full model fits.
- Styler Highlighting: Exclusively applies styling to the pandas summary table to highlight
  the best-performing model (based on highest R-squared score) using a soft green color.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Import backend utility engines from core/ package
from core.data_handler import load_custom_data, load_fallback_data
from core.preprocessor import clean_and_scale_data
from core.ml_engine import run_regression_benchmark

def clear_reg_state():
    """Callback to purge cached models when configuration changes."""
    for key in list(st.session_state.keys()):
        if key.startswith("reg_model_cache_"):
            del st.session_state[key]

def show_regression_workspace():
    """Renders the Regression Workspace views and interactive widgets."""
    st.markdown("## 📈 Regression Workspace")
    
    # -----------------------------------------
    # Sidebar Configuration Options
    # -----------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📁 Regression Data Source")
    
    # Set data source
    reg_source = st.sidebar.radio(
        "Select dataset type:",
        ["Built-in Fallback", "Upload CSV/Excel"],
        key="reg_source_radio",
        on_change=clear_reg_state
    )

    uploaded_file = None
    if reg_source == "Upload CSV/Excel":
        st.sidebar.markdown("**Dataset required**")
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV or Excel dataset",
            type=["csv", "xlsx", "xls"],
            key="reg_file_uploader",
            on_change=clear_reg_state
        )
        if uploaded_file is None:
            st.sidebar.warning("Upload a dataset to continue.")
    
    # Advanced Model Settings
    st.sidebar.markdown("## ⚙️ Model Parameters")
    train_size = st.sidebar.slider(
        "Train Split Ratio", 
        min_value=0.5, 
        max_value=0.9, 
        value=0.8, 
        step=0.05, 
        key="reg_train_ratio",
        on_change=clear_reg_state
    )
    random_state = st.sidebar.number_input(
        "Random Seed", 
        min_value=1, 
        max_value=9999, 
        value=42, 
        step=1, 
        key="reg_seed",
        on_change=clear_reg_state
    )
    max_rows = st.sidebar.slider(
        "Max Rows (Downsample Limit)", 
        min_value=500, 
        max_value=10000, 
        value=2000, 
        step=500, 
        key="reg_row_limit",
        on_change=clear_reg_state
    )
    
    df = None
    fallback_target = None
    
    # Handle Data Loading
    if reg_source == "Upload CSV/Excel":
        if uploaded_file is not None:
            try:
                # Load custom uploaded dataset via core engine
                df = load_custom_data(uploaded_file, uploaded_file.name)
            except Exception as e:
                st.error(f"Error loading uploaded file: {e}")
        else:
            st.warning("A CSV or Excel dataset is required to continue in upload mode.")
    else:
        # Load built-in fallback dataset (California Housing or Diabetes) via core engine
        df, fallback_target = load_fallback_data("california_housing")
        st.sidebar.success("ℹ️ Built-in dataset (California Housing) active.")
        
    if df is not None:
        # -----------------------------------------
        # Preprocessing & Target Selection UI
        # -----------------------------------------
        cols = df.columns.tolist()
        default_target = fallback_target if fallback_target in cols else cols[-1]
        default_idx = cols.index(default_target) if default_target in cols else len(cols) - 1
        
        target_col = st.selectbox(
            "🎯 Select Target Variable (Y):",
            options=cols,
            index=default_idx,
            key="reg_target_select",
            on_change=clear_reg_state
        )
        
        # Issue warning if selected target is categorical (since this is regression)
        if not np.issubdtype(df[target_col].dtype, np.number):
            st.warning("⚠️ Warning: The selected target variable is non-numeric. Regression is best suited for continuous numbers. Target encoding will be applied automatically.")
            
        # Display dataset statistics overview
        st.markdown(f"""
        <div class="section-card" style="background-color: var(--secondary-background-color); border: 1px solid rgba(128,128,128,0.15); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;">
            <h4 style="margin-top: 0;">📊 Dataset Overview</h4>
            <strong>Original Rows:</strong> {df.shape[0]} | <strong>Columns:</strong> {df.shape[1]} <br>
            <strong>Training Rows (after downsample limit):</strong> {min(df.shape[0], max_rows)}
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🔍 View Raw Dataset Sample"):
            st.dataframe(df.head(10), use_container_width=True)
            
        # Generate unique cache state key based on config parameters
        reg_state_key = f"reg_model_cache_{reg_source}_{target_col}_{train_size}_{random_state}_{max_rows}"
        
        # Fit models or fetch them from cache to optimize playground interaction
        if reg_state_key not in st.session_state:
            with st.spinner("Executing core ML benchmark engine (training exactly 6 models)..."):
                # Subsample if dataset is too large to maintain responsiveness
                if len(df) > max_rows:
                    df_sample = df.sample(n=max_rows, random_state=random_state)
                else:
                    df_sample = df.copy()
                    
                # 1. Clean, scale, impute, and encode via core preprocessor
                (
                    X_train, 
                    X_test, 
                    y_train, 
                    y_test, 
                    scaler, 
                    feature_encoders, 
                    target_encoder, 
                    X_original, 
                    num_cols, 
                    cat_cols
                ) = clean_and_scale_data(
                    df_sample, 
                    target_col, 
                    train_size=train_size, 
                    random_state=random_state
                )
                
                # 2. Train models and generate metrics table via core ml_engine
                results_df, trained_models = run_regression_benchmark(
                    X_train, 
                    X_test, 
                    y_train, 
                    y_test, 
                    random_state=random_state
                )
                
                # Cache results in streamlit session state
                st.session_state[reg_state_key] = {
                    'results_df': results_df,
                    'trained_models': trained_models,
                    'scaler': scaler,
                    'feature_encoders': feature_encoders,
                    'target_encoder': target_encoder,
                    'X_original': X_original,
                    'num_cols': num_cols,
                    'cat_cols': cat_cols
                }
                
        # Load objects from session state cache
        cached = st.session_state[reg_state_key]
        results_df = cached['results_df']
        trained_models = cached['trained_models']
        scaler = cached['scaler']
        feature_encoders = cached['feature_encoders']
        target_encoder = cached['target_encoder']
        X_original = cached['X_original']
        num_cols = cached['num_cols']
        cat_cols = cached['cat_cols']
        
        if not results_df.empty:
            # -----------------------------------------
            # Performance Table & Dynamic Highlighting
            # -----------------------------------------
            st.markdown("### 🏆 Benchmark Results")
            best_model_name = results_df['R-squared (R2)'].idxmax()
            best_r2 = results_df.loc[best_model_name, 'R-squared (R2)']
            
            # Winner alert card
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.15) 100%); border-left: 6px solid #10B981; padding: 1.25rem; border-radius: 8px; margin-bottom: 1.5rem;">
                <strong>🏆 Winning Model:</strong> <span style="font-size: 1.1rem; color: #10B981; font-weight: 700;">{best_model_name}</span> 
                with an R² Score of <strong>{best_r2:.4f}</strong>.
            </div>
            """, unsafe_allow_html=True)
            
            # Stylize dataframe row dynamically for the winning model (soft green highlight)
            def highlight_winner(row):
                color = 'background-color: rgba(16, 185, 129, 0.18); font-weight: 700; border-left: 5px solid #10B981;'
                return [color if row.name == best_model_name else '' for _ in row]
                
            styled_df = results_df.style.apply(highlight_winner, axis=1).format("{:.4f}")
            st.dataframe(styled_df, use_container_width=True)
            
            # -----------------------------------------
            # Visual comparison (Plotly Charts)
            # -----------------------------------------
            st.markdown("### 📊 Visual Comparisons")
            tab_r2, tab_errors = st.tabs(["🎯 Primary Metric (R² Score)", "📉 Error Variance (MAE & RMSE)"])
            
            with tab_r2:
                fig_r2 = px.bar(
                    results_df.reset_index(),
                    x='Model',
                    y='R-squared (R2)',
                    color='R-squared (R2)',
                    color_continuous_scale='Teal',
                    title='Model Comparison: R-squared (R2) Score (Higher is Better)',
                    labels={'R-squared (R2)': 'R2 Score', 'Model': 'Estimator Name'}
                )
                fig_r2.update_layout(xaxis_tickangle=-15, coloraxis_showscale=False, hovermode="x unified")
                st.plotly_chart(fig_r2, use_container_width=True)
                
            with tab_errors:
                melted = results_df.reset_index().melt(
                    id_vars='Model',
                    value_vars=['Mean Absolute Error (MAE)', 'Root Mean Squared Error (RMSE)'],
                    var_name='Metric',
                    value_name='Error Value'
                )
                fig_err = px.bar(
                    melted,
                    x='Model',
                    y='Error Value',
                    color='Metric',
                    barmode='group',
                    color_discrete_map={
                        'Mean Absolute Error (MAE)': '#3B82F6',
                        'Root Mean Squared Error (RMSE)': '#EF4444'
                    },
                    title='Model Comparison: MAE and RMSE Error (Lower is Better)'
                )
                fig_err.update_layout(xaxis_tickangle=-15, hovermode="x unified")
                st.plotly_chart(fig_err, use_container_width=True)
                
            # -----------------------------------------
            # Real-Time Prediction Playground
            # -----------------------------------------
            st.markdown("---")
            st.markdown("### 🔮 Real-Time Prediction Playground")
            
            # Sub-sample features displayed if dataset is extremely wide
            if len(X_original.columns) > 12:
                st.info("💡 Note: The dataset has more than 12 features. The Prediction Playground displays inputs for the first 10 features, and the remaining features are set to their training mean/mode.")
                cols_to_show = list(X_original.columns)[:10]
            else:
                cols_to_show = list(X_original.columns)
                
            # Render input widgets in 3-columns layout grid
            num_cols_grid = 3
            grid = st.columns(num_cols_grid)
            
            input_data = {}
            for idx, col in enumerate(cols_to_show):
                with grid[idx % num_cols_grid]:
                    if col in cat_cols:
                        options = list(X_original[col].unique())
                        val = st.selectbox(f"Select {col}", options=options, key=f"pred_reg_{col}")
                        input_data[col] = val
                    else:
                        min_v = float(X_original[col].min())
                        max_v = float(X_original[col].max())
                        mean_v = float(X_original[col].mean())
                        
                        if min_v == max_v:
                            val = st.number_input(f"{col} (constant)", value=mean_v, disabled=True, key=f"pred_reg_{col}")
                        else:
                            step = 1.0 if (max_v - min_v) > 20 else 0.1
                            val = st.slider(f"{col}", min_value=min_v, max_value=max_v, value=mean_v, step=step, key=f"pred_reg_{col}")
                        input_data[col] = val
                        
            # Reconstruct the single input vector using mean/mode for unshown columns
            full_input = {}
            for col in X_original.columns:
                if col in cols_to_show:
                    full_input[col] = input_data[col]
                else:
                    if col in cat_cols:
                        full_input[col] = X_original[col].mode()[0]
                    else:
                        full_input[col] = X_original[col].mean()
                        
            # Map input row into a DataFrame and encode/scale
            input_df = pd.DataFrame([full_input])
            for col in cat_cols:
                le = feature_encoders[col]
                try:
                    input_df[col] = le.transform([str(input_df[col].iloc[0])])
                except Exception:
                    input_df[col] = 0
                    
            input_scaled = scaler.transform(input_df)
            
            # Predict using the cached best model
            best_model = trained_models[best_model_name]
            pred = best_model.predict(input_scaled)[0]
            
            # Display Prediction Output & context distribution plot
            st.markdown("#### Prediction Output")
            pred_col1, pred_col2 = st.columns([1, 2])
            
            with pred_col1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.15) 100%); border: 1px solid rgba(59, 130, 246, 0.3); padding: 1.5rem; border-radius: 10px; text-align: center; margin-top: 1rem; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.05);">
                    <div style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.7;">Predicted Output ({target_col})</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #3B82F6; margin-top: 0.25rem;">{pred:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with pred_col2:
                # Plotly distribution comparison chart
                y_train_vals = df[target_col]
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=y_train_vals,
                    name='Training Target Distribution',
                    marker_color='#3B82F6',
                    opacity=0.6,
                    nbinsx=35
                ))
                fig_dist.add_vline(
                    x=pred, 
                    line_width=3, 
                    line_dash="dash", 
                    line_color="#EF4444",
                    annotation_text="Prediction",
                    annotation_position="top right"
                )
                fig_dist.update_layout(
                    margin=dict(l=10, r=10, t=20, b=10),
                    height=180,
                    showlegend=False,
                    xaxis_title=f"{target_col} Values",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': False})
