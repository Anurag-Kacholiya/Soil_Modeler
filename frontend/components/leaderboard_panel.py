import streamlit as st
import pandas as pd

def render_leaderboard_panel(df: pd.DataFrame):
    if df is None or df.empty:
        st.caption("No models available.")
        return

    # Always use deduplicated model list
    df = df.drop_duplicates(subset=["model_id"], keep="last")

    # Ranking by R² score
    df_sorted = df.sort_values("r2", ascending=False).reset_index(drop=True)

    selected_id = st.session_state.get("selected_model_id")
    updated_id = st.session_state.get("just_updated_model")
    movement = st.session_state.get("rank_movement")
    r2_delta = st.session_state.get("r2_delta")

    st.markdown("""
        <style>
        .leaderboard-wrapper {
            padding: 5px;
            border-radius: 10px;
            margin-bottom: 10px;
        }

        .selectedGlow {
            border: 2px solid #3B82F6 !important;
            box-shadow: 0 0 10px #3B82F6;
            background-color: #0A3C2F !important;
        }

        .updatedGlow {
            border: 2px solid #34D399 !important;
            box-shadow: 0 0 15px #34D399;
            background-color: #064E3B !important;
        }

        div.stButton > button {
            background-color: #0B241E;
            color: #ECFDF5;
            border: 1px solid #113028;
            border-radius: 8px;
            padding: 12px 16px;
            width: 100%;
            text-align: left;
            font-size: 0.9rem;
            line-height: 1.45;
            transition: all 0.25s ease;
        }

        div.stButton > button:hover {
            background-color: #0F3B34;
            border-color: #34D399;
            transform: translateX(4px);
        }

        .badge {
            font-size: 0.7rem;
            padding: 2px 6px;
            border-radius: 8px;
            margin-bottom: 4px;
            font-weight: bold;
            display: inline-block;
        }

        .selectedBadge { background-color: #3B82F6; color: white; }
        .updatedBadge { background-color: #34D399; color: #02120F; }
        .upgradeBadge { background-color: #34D399; color: #02120F; }
        .droppedBadge { background-color: #F87171; color: #02120F; }
        .stableBadge { background-color: #FCD34D; color: #02120F; }
        </style>
    """, unsafe_allow_html=True)

    for idx, row in df_sorted.iterrows():
        model_id = row["model_id"]
        rank = idx + 1

        if rank == 1: icon = "🥇"
        elif rank == 2: icon = "🥈"
        elif rank == 3: icon = "🥉"
        else: icon = f"Rank {rank}"

        # Movement Badge Logic
        badge_line = ""
        if updated_id == model_id and movement is not None:
            if movement == "up":
                badge_line = f"<div class='badge upgradeBadge'>⬆ Upgraded (+{r2_delta:.3f} R²)</div>"
            elif movement == "down":
                badge_line = f"<div class='badge droppedBadge'>⬇ Dropped ({r2_delta:.3f} R²)</div>"
            else:
                badge_line = "<div class='badge updatedBadge'>➖ No Change</div>"


        # Selected Badge
        if selected_id == model_id:
            badge_line += " <div class='badge selectedBadge'>SELECTED</div>"

        # Glow highlight
        if updated_id == model_id:
            glow_class = "leaderboard-wrapper updatedGlow"
        elif selected_id == model_id:
            glow_class = "leaderboard-wrapper selectedGlow"
        else:
            glow_class = "leaderboard-wrapper"

        # Show movement badge on top
        if badge_line:
            st.markdown(f"<div style='margin-left:4px;'>{badge_line}</div>", unsafe_allow_html=True)

        # Button label
        label = f"{icon}  {row['model'].upper()}  |  {row['preprocessing'].upper()}"

        # Unique key MUST include rank to avoid duplicates
        btn_key = f"{model_id}_{rank}"

        # Card wrapper
        st.markdown(f"<div class='{glow_class}' style='padding:4px;'>", unsafe_allow_html=True)

        if st.button(label, key=btn_key, use_container_width=True):
            st.session_state.selected_model_id = model_id
            st.session_state.selected_model_data = row.to_dict()
            st.session_state.old_rank = None
            st.session_state.rank_movement = None
            st.session_state.r2_delta = None
            st.session_state.just_updated_model = None
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
