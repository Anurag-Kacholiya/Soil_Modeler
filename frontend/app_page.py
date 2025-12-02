import streamlit as st
from backend.main import load_leaderboard
from frontend.components.leaderboard_panel import render_leaderboard_panel
from frontend.components.center_panel import render_center_panel
from frontend.components.right_panel import render_raw_spectra, render_band_accuracy

def show_app_page():
    # Set sidebar to expanded so it's visible by default
    st.set_page_config(page_title="Spectral Soil Modeler", layout="wide", initial_sidebar_state="expanded")

    # --- CSS STYLING ---
    st.markdown("""
        <style>
        /* 1. MAIN BACKGROUND */
        .stApp {
            background-color: #051F1A;
            color: #ECFDF5;
        }
        
        .block-container {
            padding-top: 2rem; /* Give space for the header */
            padding-bottom: 1rem;
            max-width: 100%;
        }

        /* 2. HEADER & SIDEBAR TOGGLE FIX */
        /* Instead of hiding the header, we style it to match the theme */
        header[data-testid="stHeader"] {
            background-color: #051F1A; /* Match app background */
            border-bottom: 1px solid #134E4A; /* Subtle border */
        }
        
        /* Color the Open/Close Sidebar Button (The Arrow) */
        button[kind="header"] {
            color: #34D399 !important; /* Bright Green Arrow */
            background-color: transparent !important;
        }
        button[kind="header"]:hover {
            color: #ECFDF5 !important;
            background-color: #0F3B34 !important;
        }

        /* 3. SIDEBAR STYLING */
        section[data-testid="stSidebar"] {
            background-color: #02120F; /* Darker than main bg */
            border-right: 1px solid #134E4A;
        }
        
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3 {
            color: #6EE7B7 !important;
            font-size: 1.2rem !important;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            border-bottom: 1px solid #134E4A;
            padding-bottom: 1rem;
        }

        /* 4. MAIN PANEL COLUMNS (Resizable Wrapper) */
        div[data-testid="column"] {
            resize: horizontal;
            overflow: auto !important;
            min-width: 350px;
            flex: 1 1 auto !important;
            
            /* Visual Separator */
            border-right: 2px solid #0F3B34;
            padding-right: 15px;
            transition: border-color 0.2s;
        }
        
        div[data-testid="column"]::-webkit-scrollbar { width: 0px; height: 0px; }

        /* 5. SCROLLABLE CONTENT BOXES */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #0B2B26;
            border: 1px solid #134E4A;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        }
        
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: #34D399;
        }

        /* 6. TYPOGRAPHY (Panel Headers) */
        h3 {
            position: sticky;
            top: 0;
            background-color: #0B2B26;
            z-index: 50;
            padding-top: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid #134E4A;
            margin-top: 0 !important;
            color: #6EE7B7 !important;
            text-transform: uppercase;
            font-size: 1.1rem !important;
        }

        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #02120F; }
        ::-webkit-scrollbar-thumb { background: #134E4A; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #34D399; }
        
        .js-plotly-plot .plotly .main-svg, .js-plotly-plot .plotly {
            background: rgba(0,0,0,0) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- DATA LOADING ---
    if "leaderboard" not in st.session_state:
        st.session_state.leaderboard = load_leaderboard()

    df = st.session_state.leaderboard

    # --- SIDEBAR: LEADERBOARD ---
    with st.sidebar:
        st.markdown("### 🏆 Leaderboard")
        
        # Navigation
        if st.button("⬅ Back to Home", use_container_width=True):
            st.session_state.page = "landing"
            st.session_state.selected_model_id = None
            st.session_state.selected_model_data = None
            st.session_state.just_updated_model = None
            st.session_state.rank_movement = None
            st.session_state.r2_delta = None
            st.rerun()
            
        st.markdown("---")
        
        # Scrollable container inside sidebar
        with st.container(height=750):
            render_leaderboard_panel(df)

    # --- MAIN PAGE: 2 PANELS ---
    col_analysis, col_diagnostics = st.columns([2, 1.2], gap="medium")

    # PANEL 1: MODEL ANALYSIS (Center)
    with col_analysis:
        with st.container(height=850):
            st.markdown("### MODEL RESULTS")
            render_center_panel()

    # PANEL 2: DIAGNOSTICS (Right)
    with col_diagnostics:
        with st.container(height=850):
            st.markdown("### Diagnostics")
            
            st.markdown("#### Raw Spectra")
            render_raw_spectra()
            
            st.markdown("<div style='height: 30px'></div>", unsafe_allow_html=True)
            
            st.markdown("#### Band Sensitivity")
            render_band_accuracy()