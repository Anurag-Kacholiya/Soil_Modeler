import streamlit as st
import pandas as pd
from pathlib import Path
from backend.main import run_full_pipeline, get_available_datasets

DATA_DIR = Path('dataset')
LEADERBOARD_FILE = Path('leaderboard.json')

def show_landing_page():
    st.title("The Spectral Soil Modeler")
    st.caption("Automated ML pipeline for spectral soil property prediction")

    col1, col2, col3 = st.columns(3)
    available = get_available_datasets()
    if not available:
        st.warning("No .xls/.xlsx/.csv datasets found in 'dataset'. Add files and refresh.")
        return

    with col1:
        selected_dataset = st.selectbox("Select Dataset", available)

    # ✅ Always read CSV (no Excel check, no numeric-only filter)
    with col2:
        try:
            df_peek = pd.read_csv(DATA_DIR / selected_dataset, nrows=1)
            all_columns = list(df_peek.columns)
        except Exception as e:
            st.error(f"Error reading dataset: {e}")
            all_columns = []

        if not all_columns:
            st.warning("No columns found in the selected file.")
            selected_column = None
        else:
            selected_column = st.selectbox("Select Target Column", all_columns)

    with col3:
        property_label = st.selectbox(
            "Assumed Property",
            ['Clay', 'Sand', 'Silt', 'TOC', 'Moisture', 'Other']
        )

    run_disabled = not (selected_dataset and selected_column and property_label)
    if st.button("🚀 Run Full Analysis", disabled=run_disabled):
        st.session_state.selected_dataset = selected_dataset
        st.session_state.selected_column_name = selected_column
        st.session_state.selected_target = property_label

        leaderboard_df = run_full_pipeline(selected_dataset, selected_column, property_label)
        if not leaderboard_df.empty:
            st.session_state.leaderboard = leaderboard_df
            st.session_state.page = 'results'
            st.rerun()
        else:
            st.error("Pipeline failed. Check dataset and try again.")