import streamlit as st
import pandas as pd
import numpy as np
import joblib

from backend.main import (
    run_single_pipeline,
    plot_scatter_interactive,
    plot_feature_importance_interactive,
    plot_wavelength_variance_interactive,
    plot_property_correlation_interactive,
    plot_spectral_profiles_interactive,
    load_data, preprocess_data
)
from models import MODEL_CONFIG


def render_center_panel():
    selected = st.session_state.get("selected_model_data")

    # DEFAULT EMPTY PANEL
    if selected is None:
        st.markdown("""
            <div style="text-align: center; padding: 4rem; color: #6EA89E; opacity: 0.7;">
                <h3>👈 Select a Model</h3>
                <p>Pick a model from the leaderboard to continue</p>
            </div>
        """, unsafe_allow_html=True)
        return

    # LOAD MODEL + DATA
    model = joblib.load(selected["pickle_path"])
    y_true = np.load(selected["y_true_path"])
    y_pred = np.load(selected["y_pred_path"])
    X, y_data = load_data(selected["dataset"], selected["target_column"])
    X_prep = preprocess_data(X, selected["preprocessing_key"])

    st.session_state.X_raw = X
    st.session_state.X_prep = X_prep
    st.session_state.y_data = y_data
    st.session_state.band_names = list(X.columns)
    st.session_state.loaded_model = model

    # SUMMARY HEADER
    st.markdown(f"""
        <div style="background: rgba(19, 78, 74, 0.5); padding: 15px; border-radius: 8px; border-left: 5px solid #34D399; margin-bottom: 20px;">
            <h2 style="margin:0; color: #ECFDF5; font-size: 1.5rem;">{selected['model'].upper()}</h2>
            <p style="margin:0; color: #A7F3D0; font-size: 0.9rem; font-family: monospace;">
                DATASET: {selected['dataset']} | TARGET: {selected['target']} | PREPROCESS: {selected['preprocessing']}
            </p>
        </div>
    """, unsafe_allow_html=True)

    # METRICS
    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{selected['r2']:.4f}")
    c2.metric("RMSE", f"{selected['rmse']:.4f}")
    c3.metric("RPD", f"{selected['rpd']:.4f}")

    st.markdown("---")

    # TABS
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "📈 Visualizations", "⚙ Hyperparameters", "🔁 Retrain"])

    # TAB 1: PERFORMANCE
    with tab1:
        st.caption("Predicted vs Measured Values")
        st.plotly_chart(plot_scatter_interactive(y_true, y_pred), use_container_width=True)

    # TAB 2
    with tab2:
        st.markdown("#### Feature Importance")
        st.plotly_chart(
            plot_feature_importance_interactive(model, X_prep, y_data, st.session_state.band_names),
            use_container_width=True
        )

        st.markdown("---")

        a, b = st.columns(2)
        with a:
            st.markdown("#### Band Variance")
            st.plotly_chart(plot_wavelength_variance_interactive(X_prep, st.session_state.band_names), use_container_width=True)
        with b:
            st.markdown("#### Property Correlation")
            st.plotly_chart(plot_property_correlation_interactive(X_prep, y_data, st.session_state.band_names), use_container_width=True)

        st.markdown("---")
        st.markdown("#### Spectral Profiles")
        st.plotly_chart(plot_spectral_profiles_interactive(
            X_prep,
            band_names=st.session_state.band_names,
            preprocess_key=selected["preprocessing_key"]
        ),use_container_width=True)

    # TAB 3: Hyperparameters
    with tab3:
        st.markdown("#### Model Configuration")
        df_hp = pd.DataFrame(selected["hyperparameters"], index=[0]).T
        df_hp.columns = ["Value"]
        st.dataframe(df_hp, use_container_width=True)

    # TAB 4: RETRAIN
    with tab4:
        st.markdown("### 🔁 Retrain Model")

        model_name = selected["model"]
        hp_schema = MODEL_CONFIG.get(model_name, {}).get("params", [])

        # Reset editable params when changing model
        if st.session_state.get("live_model_id") != selected["model_id"]:
            st.session_state.live_model_id = selected["model_id"]
            st.session_state.edited_params = dict(selected["hyperparameters"])

        new_params = st.session_state.edited_params

        st.markdown("#### Edit Parameters")

        for param in hp_schema:
            name = param["name"]
            ptype = param["type"]
            default = new_params.get(name, param.get("default"))

            if ptype == "int":
                val = st.number_input(name, value=int(default), min_value=param["min"], max_value=param["max"], step=1)
            elif ptype == "float":
                val = st.number_input(name, value=float(default), min_value=param["min"], max_value=param["max"], step=0.01)
            elif ptype == "select":
                options = param["options"]
                if default not in options:
                    default = options[0]
                val = st.selectbox(name, options, index=options.index(default))
            else:
                val = default

            new_params[name] = val

        st.markdown("---")

        if st.button("🚀 Retrain Model", use_container_width=True):
            with st.status("Training new model...", expanded=True):

                valid_keys = [p["name"] for p in hp_schema]
                filtered_params = {k: v for k, v in new_params.items() if k in valid_keys}

                result = run_single_pipeline(
                    dataset_name=selected["dataset"],
                    target_column_name=selected["target_column"],
                    target_property_label=selected["target"],
                    prep_key=selected["preprocessing_key"],
                    model_name=model_name,
                    hyperparameters=filtered_params
                )

                if result is None:
                    st.error("❌ Training failed")
                else:
                    df = st.session_state.leaderboard.copy()

                    # old R2 and movement direction
                    old_r2 = selected["r2"]
                    new_r2 = result["r2"]
                    r2_delta = new_r2 - old_r2

                    if r2_delta > 0:
                        movement = "up"
                    elif r2_delta < 0:
                        movement = "down"
                    else:
                        movement = "none"

                    # replace old model
                    df = df[df["model_id"] != selected["model_id"]]
                    df.loc[len(df)] = result

                    # update session
                    st.session_state.leaderboard = df.sort_values("r2", ascending=False)
                    st.session_state.selected_model_id = result["model_id"]
                    st.session_state.selected_model_data = result
                    st.session_state.just_updated_model = result["model_id"]
                    st.session_state.rank_movement = movement
                    st.session_state.r2_delta = r2_delta

                    st.success(f"🎉 Retraining Done! ΔR² = {r2_delta:+.4f}")
                    st.rerun()