import streamlit as st
import pandas as pd
from pathlib import Path

# Backend imports
from backend.main import get_available_datasets, run_full_pipeline, load_leaderboard

DATA_DIR = Path("dataset")

def show_landing_page():
    # Only set page config if it hasn't been set
    try:
        st.set_page_config(page_title="Spectral Soil Modeler", layout="wide")
    except:
        pass

    # ---- CSS STYLING (Dark Green + Glassmorphism) ----
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        .stApp {
            background: radial-gradient(circle at 50% 10%, #0f3832 0%, #051a17 60%, #02120F 100%);
            font-family: "Inter", sans-serif;
            color: #ECFDF5;
        }

        /* TYPOGRAPHY */
        h1 {
            font-weight: 800;
            background: linear-gradient(90deg, #FFFFFF, #6EE7B7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.5rem !important;
            margin-bottom: 0rem;
            letter-spacing: -0.03em;
        }
        
        h3 {
            color: #34D399 !important;
            font-weight: 600;
            margin-top: 0 !important;
        }

        .subtitle {
            color: #A7F3D0;
            font-size: 1.25rem;
            font-weight: 300;
            margin-bottom: 2rem;
            opacity: 0.9;
        }

        /* INFO BOX (LEFT SIDE) */
        .info-box {
            padding: 2rem 0;
        }
        .step-badge {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #6EE7B7;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-right: 10px;
        }
        .highlight-text {
            color: #34D399;
            font-weight: 600;
        }

        /* GLASS CARD CONTAINER (RIGHT SIDE) */
        .glass-card {
            background: rgba(11, 43, 38, 0.5);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(52, 211, 153, 0.15);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
            border-radius: 24px;
            padding: 2.5rem;
        }

        /* INPUT STYLES */
        .input-label {
            font-size: 0.9rem;
            font-weight: 600;
            color: #D1FAE5;
            margin-bottom: 0.4rem;
            display: flex;
            align-items: center;
            gap: 8px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .input-helper {
            font-size: 0.8rem;
            color: #6EA89E;
            margin-bottom: 0.5rem;
        }

        /* BUTTONS */
        div.stButton > button {
            width: 100%;
            border-radius: 12px;
            padding: 0.85rem 1rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.2s ease-in-out;
            border: 1px solid transparent;
        }

        /* Primary Button */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #059669 0%, #047857 100%);
            color: #FFFFFF !important;
            box-shadow: 0 4px 20px rgba(5, 150, 105, 0.3);
        }
        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-2px);
            background: linear-gradient(135deg, #10B981 0%, #059669 100%);
            box-shadow: 0 6px 25px rgba(16, 185, 129, 0.4);
            border-color: #34D399;
        }
        div.stButton > button[kind="primary"]:disabled {
            background: #0f2e28;
            color: #3f5e58 !important;
            box-shadow: none;
            border-color: #1a423b;
        }

        /* Secondary Button */
        div.stButton > button[kind="secondary"] {
            background: rgba(6, 78, 59, 0.3);
            border: 1px solid #059669;
            color: #6EE7B7 !important;
        }
        div.stButton > button[kind="secondary"]:hover {
            background: rgba(6, 78, 59, 0.6);
            color: #FFFFFF !important;
            border-color: #34D399;
        }
        
        /* Status Box */
        .status-console {
            background: rgba(0, 0, 0, 0.4);
            border-left: 4px solid #10B981;
            padding: 16px;
            border-radius: 0 8px 8px 0;
            margin-top: 24px;
            margin-bottom: 24px;
            font-family: 'Courier New', monospace;
        }
        </style>
    """, unsafe_allow_html=True)

    # ---- LAYOUT: HEADER ----
    st.markdown("<h1>The Spectral Soil Modeler</h1>", unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Automated Pipeline for Soil Property Prediction using Hyperspectral Reflectance</div>', unsafe_allow_html=True)
    
    st.markdown("<div style='height: 40px'></div>", unsafe_allow_html=True)

    # ---- SPLIT LAYOUT ----
    # Left: Informational Text | Right: Functional Card
    col_text, col_spacer, col_card = st.columns([1.2, 0.2, 1])

    with col_text:
        st.markdown(
            """
            ### 🧠 What Does This Tool Do?

            This tool automatically trains machine learning models on **hyperspectral soil datasets**  
            to predict real soil properties.

            It tests five different regression models:
            - :green[PLSR] — Partial Least Squares
            - :green[Cubist] — Rule-based Regression
            - :green[GBRT] — Gradient Boosted Trees
            - :green[KRR] — Kernel Ridge Regression
            - :green[SVR] — Support Vector Regression

            Each model is trained using three preprocessing methods:
            - **Reflectance**
            - **Absorbance (Log 1/R)**
            - **Continuum Removal**

            Models are compared using:
            - :orange[R²] (Accuracy)
            - :blue[RMSE] (Error)
            - :green[RPD] (Prediction Strength)

            ---
            ### 🧾 Steps to Begin

            :green-badge[Step 1] Select the dataset (.csv file containing spectral bands + soil values)  
            :green-badge[Step 2] Select the target column (actual soil property values)  
            :green-badge[Step 3] Choose the soil property label for the experiment  

            ---
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)


    with col_card:
        
        datasets = get_available_datasets()
        
        if not datasets:
            st.error("⚠️ No datasets found in 'dataset' folder.")
        else:
            # 1. Dataset Selection
            st.markdown('<div class="input-label">📂 Data Source</div>', unsafe_allow_html=True)
            selected_dataset = st.selectbox("Dataset", datasets, label_visibility="collapsed")
            
            st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

            # 2. Target Column
            st.markdown('<div class="input-label">🎯 Target Variable</div>', unsafe_allow_html=True)
            try:
                # Read only first row to get columns quickly
                df_peek = pd.read_csv(DATA_DIR / selected_dataset, nrows=1)
                numeric_cols = [c for c in df_peek.columns if df_peek[c].dtype.kind in 'biufc']
                all_columns = numeric_cols if numeric_cols else list(df_peek.columns)
                selected_column = st.selectbox("Target", all_columns, label_visibility="collapsed")
            except Exception as e:
                st.error(f"Error reading file")
                selected_column = None

            st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

            # 3. Property Label
            st.markdown('<div class="input-label">🧪 Soil Property</div>', unsafe_allow_html=True)
            soil_properties = ["Clay", "Sand", "Silt", "TOC", "Moisture", "pH", "Nitrogen", "Other"]
            property_label = st.selectbox("Property", soil_properties, label_visibility="collapsed")

            st.markdown("<hr style='border-color: rgba(52, 211, 153, 0.2); margin: 2rem 0;'>", unsafe_allow_html=True)

            # ---- ACTION BUTTONS ----
            run_disabled = not (selected_dataset and selected_column and property_label)

            if st.button("🚀 RUN PIPELINE", type="primary", disabled=run_disabled, use_container_width=True):
                # UI Status Components
                status_placeholder = st.empty()
                progress_bar = st.progress(0)

                def progress_callback(model_name, prep_key):
                    status_placeholder.markdown(
                        f"""
                        <div class="status-console">
                            <h4 style="color:#76b852; margin:0 0 5px 0;">⚡ Training Active</h4>
                            <p style="color:#e0e0e0; font-size:0.9rem; margin:0;">
                                <b>Model:</b> <span style="color:#8DC26F;">{model_name}</span><br>
                                <b>Mode:</b> <span style="color:#4FC3F7;">{prep_key}</span>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                def update_progress(progress_ratio):
                    progress_bar.progress(int(progress_ratio * 100))

                try:
                    leaderboard_df = run_full_pipeline(
                        selected_dataset,
                        selected_column,
                        property_label,
                        progress_callback=progress_callback,
                        update_progress=update_progress,
                    )
                    
                    progress_bar.progress(100)
                    status_placeholder.markdown(
                        """
                        <div class="status-console" style="border-left-color: #34D399;">
                            <h4 style="color:#34D399; margin:0;">🎉 Analysis Complete!</h4>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    st.session_state.leaderboard = leaderboard_df
                    st.session_state.page = "results"
                    st.rerun()

                except Exception as e:
                    st.error(f"Pipeline failed: {str(e)}")

            st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)

            st.markdown("### Recently Trained Models")

            saved = load_leaderboard()

            if saved is None or saved.empty:
                st.warning("No training history found.")
            else:
                # Create unique experiment signature
                saved["unique_key"] = saved["dataset"] + "_" + saved["target"]

                # Remove duplicates, keep latest occurrence
                deduped = saved.drop_duplicates(subset=["unique_key"], keep="last")

                # Last 5 experiments (sorted newest first)
                recent = deduped.tail(5).iloc[::-1]

                for _, row in recent.iterrows():
                    label = (
                        f"📁 **{row['dataset']}**  \n"
                        f"🧪 Property: **{row['target']}**"
                    )

                    # Safe unique key for button
                    key_unique = f"recent_{row['unique_key']}"

                    if st.button(label, key=key_unique, use_container_width=True):

                        # Load full results related to this experiment
                        filtered = saved[
                            (saved["dataset"] == row["dataset"]) &
                            (saved["target"] == row["target"])
                        ].copy()

                        st.session_state.leaderboard = filtered
                        st.session_state.selected_model_id = row["model_id"]
                        st.session_state.just_updated_model = None
                        st.session_state.rank_movement = None

                        st.session_state.page = "results"
                        st.rerun()


        st.markdown('</div>', unsafe_allow_html=True)
