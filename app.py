# app.py

import streamlit as st

st.set_page_config(page_title="Spectral Soil Modeler", layout="wide", page_icon="🌱")

from frontend.landing_page import show_landing_page
from frontend.app_page import show_app_page
from backend.main import load_leaderboard


def main():
    # Initialize session state variables
    if "page" not in st.session_state:
        st.session_state.page = "landing"

    if "leaderboard" not in st.session_state:
        st.session_state.leaderboard = load_leaderboard()

    # Route to appropriate page
    if st.session_state.page == "landing":
        show_landing_page()
    else:
        show_app_page()


if __name__ == "__main__":
    main()