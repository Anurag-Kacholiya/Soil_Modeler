import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import traceback
from backend.main import (
    plot_scatter,
    plot_feature_importance,
    load_data,
    preprocess_data,
    plot_wavelength_variance,
    plot_property_wavelength_relationship,
    plot_spectral_profiles
)

def show_visualization_page():
    st.header("📊 Model Visualization Dashboard")

    # Back button
    if st.button("⬅️ Back to Leaderboard"):
        st.session_state.page = 'results'
        st.session_state.selected_model_data = None
        st.rerun()

    # Check if a model is selected
    if st.session_state.selected_model_data is None:
        st.error("No model selected. Returning to leaderboard.")
        time.sleep(1)
        st.session_state.page = 'results'
        st.rerun()

    data = st.session_state.selected_model_data

    # Load model and predictions
    try:
        model = joblib.load(data['pickle_path'])
        y_true = np.load(data['y_true_path'])
        y_pred = np.load(data['y_pred_path'])
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.text(traceback.format_exc())
        return

    # Display model info
    st.subheader(f"Model: {data['model']}")
    st.write(f"**Preprocessing:** {data['preprocessing']}")
    st.write(f"**Dataset:** {data['dataset']} | **Target:** {data['target']} | **Actual Column:** {data['target_column']}")
    st.divider()

    # --- Metrics + Plot ---
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Metrics (5-Fold CV)")
        st.metric("R²", f"{data['r2']:.4f}")
        st.metric("RMSE", f"{data['rmse']:.4f}")
        st.metric("RPD", f"{data['rpd']:.4f}")

        results_df = pd.DataFrame({'actual': y_true, 'predicted': y_pred})
        st.download_button(
            "💾 Download Predictions",
            data=results_df.to_csv(index=False).encode('utf-8'),
            file_name=f"{data['model_id']}_predictions.csv",
            mime='text/csv'
        )

    with col2:
        st.subheader("Predicted vs Actual (OOF)")
        try:
            fig = plot_scatter(y_true, y_pred, "Model Performance")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating scatter plot: {e}")
            st.text(traceback.format_exc())

    st.divider()

    # --- Visualization Tabs ---
    try:
        X, y_data = load_data(data['dataset'], data['target_column'])
        if X is None:
            st.error("Dataset could not be loaded for visualization.")
            return

        X_prep = preprocess_data(X, data['preprocessing_key'])
        if isinstance(X_prep, (int, float)) or X_prep is None:
            st.error("Invalid output from preprocessing.")
            return

        tabs = st.tabs([
            "Feature Importance",
            "Wavelength Variance",
            "Property-Wavelength Relationship",
            "Spectral Profiles"
        ])

        # --- Tab 1: Feature Importance ---
        with tabs[0]:
            st.subheader("Feature Importance")
            with st.spinner("Generating feature importance..."):
                try:
                    fig_imp = plot_feature_importance(model, X_prep, y_data, st.session_state.band_names)
                    st.pyplot(fig_imp)
                except Exception as e:
                    st.error(f"Error generating feature importance: {e}")
                    st.text(traceback.format_exc())

        # --- Tab 2: Wavelength Variance ---
        with tabs[1]:
            st.subheader("Wavelength Variance")
            with st.spinner("Generating variance plot..."):
                try:
                    fig_var = plot_wavelength_variance(X_prep, st.session_state.band_names)
                    st.pyplot(fig_var)
                except Exception as e:
                    st.error(f"Error generating wavelength variance: {e}")
                    st.text(traceback.format_exc())

        # --- Tab 3: Property-Wavelength Relationship ---
        with tabs[2]:
            st.subheader("Property-Wavelength Relationship")
            with st.spinner("Generating correlation plot..."):
                try:
                    fig_rel = plot_property_wavelength_relationship(X_prep, y_data, st.session_state.band_names)
                    st.pyplot(fig_rel)
                except Exception as e:
                    st.error(f"Error generating property-wavelength plot: {e}")
                    st.text(traceback.format_exc())

        # --- Tab 4: Spectral Profiles ---
        with tabs[3]:
            st.subheader("Spectral Profiles")
            with st.spinner("Generating spectral profiles..."):
                try:
                    fig_profiles = plot_spectral_profiles(X_prep, y_data, st.session_state.band_names)
                    st.pyplot(fig_profiles)
                except Exception as e:
                    st.error(f"Error generating spectral profiles: {e}")
                    st.text(traceback.format_exc())

    except Exception as e:
        st.error("❌ Could not generate visualizations.")
        st.text(traceback.format_exc())
