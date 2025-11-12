import streamlit as st
import pandas as pd
from pathlib import Path
from backend.main import run_single_pipeline
from backend.main import MODEL_CONFIG

LEADERBOARD_FILE = Path('leaderboard.json')

def show_results_page():
    """Display model leaderboard with tuning, retraining, and visualization options."""

    # --- Ensure Session Keys ---
    for k in ("selected_target", "selected_dataset", "selected_column_name", "leaderboard", "page"):
        if k not in st.session_state:
            st.session_state[k] = None

    if st.session_state.leaderboard is None:
        st.warning("No leaderboard data found. Run analysis first.")
        if st.button("⬅️ Back to Landing Page"):
            st.session_state.page = "landing"
            st.rerun()
        return

    df = st.session_state.leaderboard
    if df.empty:
        st.warning("Leaderboard is empty. Please re-run the analysis.")
        if st.button("⬅️ Back to Landing Page"):
            st.session_state.page = "landing"
            st.rerun()
        return

    # --- Header ---
    st.header(f"📊 Results for: {st.session_state.selected_target or 'Unknown'}")
    st.subheader(f"Dataset: {st.session_state.selected_dataset or 'N/A'}")

    if st.button("⬅️ Back to Landing Page"):
        st.session_state.page = "landing"
        st.rerun()

    # --- Filter the Results ---
    df_filtered = df[
        (df["dataset"] == st.session_state.selected_dataset)
        & (df["target"] == st.session_state.selected_target)
    ].copy()

    if df_filtered.empty:
        st.warning("No results found for the selected dataset/target.")
        return

    df_filtered = df_filtered.drop_duplicates(subset=["model_id"], keep="last")

    # --- Sorting ---
    sort_by = st.selectbox("Sort by:", ["r2", "rmse", "rpd"], index=0)
    ascending = True if sort_by == "rmse" else False
    df_sorted = df_filtered.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)

    # --- Show Leaderboard ---
    st.subheader("🏆 Model Leaderboard")
    st.dataframe(df_sorted[["model", "preprocessing", "r2", "rmse", "rpd"]], width="stretch")

    # --- Pagination for Models ---
    if "results_display_limit" not in st.session_state:
        st.session_state.results_display_limit = 15 # start with 15
    display_limit = st.session_state.results_display_limit
    total_results = len(df_sorted)

    st.write(f"Showing {min(display_limit, total_results)} of {total_results} models")

    # --- Fine-tune and Retrain Section ---
    st.subheader("🔧 Fine-tune, Retrain, and Visualize Models")
    st.info("Expand a model below to adjust hyperparameters, retrain, or view visualizations.")

    for idx, row in df_sorted.head(display_limit).iterrows():
        model_name = row["model"]
        prep_name = row["preprocessing"]
        r2_val = row["r2"]

        with st.expander(f"**{idx+1}. {model_name}** ({prep_name}) — R²: {r2_val:.3f}"):
            with st.form(key=f"form_{row['model_id']}"):
                st.write(f"**Fine-tune {model_name} parameters:**")
                hyperparams = {}
                model_params = MODEL_CONFIG[model_name]["params"]

                for param in model_params:
                    if param["type"] == "int":
                        hyperparams[param["name"]] = st.number_input(
                            param["name"],
                            min_value=param["min"],
                            max_value=param["max"],
                            value=row["hyperparameters"].get(param["name"], param["default"]),
                            step=1
                        )
                    elif param["type"] == "float":
                        hyperparams[param["name"]] = st.number_input(
                            param["name"],
                            min_value=param["min"],
                            max_value=param["max"],
                            value=row["hyperparameters"].get(param["name"], param["default"]),
                            format="%.3f"
                        )
                    elif param["type"] == "select":
                        options = param["options"]
                        default_idx = options.index(row["hyperparameters"].get(param["name"], param["default"]))
                        hyperparams[param["name"]] = st.selectbox(param["name"], options, index=default_idx)

                # --- Form Buttons (Side-by-side) ---
                col_form1, col_form2 = st.columns(2)
                with col_form1:
                    retrain_button = st.form_submit_button("🔁 Retrain Model")
                with col_form2:
                    visualize_button = st.form_submit_button("📊 Show Results")

            # --- Retrain Logic ---
            if retrain_button:
                with st.spinner(f"Retraining {model_name}..."):
                    new_result = run_single_pipeline(
                        dataset_name=row["dataset"],
                        target_column_name=row["target_column"],
                        target_property_label=row["target"],
                        prep_key=row["preprocessing_key"],
                        model_name=row["model"],
                        hyperparameters=hyperparams,
                    )

                if new_result:
                    st.session_state.leaderboard = pd.concat(
                        [df, pd.DataFrame([new_result])]
                    ).reset_index(drop=True)
                    st.session_state.leaderboard.to_json(
                        LEADERBOARD_FILE, orient="records", indent=2
                    )
                    st.success(f"✅ Model retrained successfully! New R²: {new_result['r2']:.3f}")
                    st.rerun()
                else:
                    st.error("⚠️ Retraining failed for this model.")

            # --- Visualization Logic ---
            if visualize_button:
                st.session_state.selected_model_data = row
                st.session_state.page = "visualize"
                st.rerun()

    # --- Pagination Buttons ---
    if display_limit < total_results:
        if st.button("🔽 Show More Results"):
            st.session_state.results_display_limit += 15
            st.rerun()
    elif total_results > 15:
        if st.button("⬆️ Show Less Results"):
            st.session_state.results_display_limit = 15
            st.rerun()
    else:
        st.caption("All results are displayed ✅")