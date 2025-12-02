import streamlit as st
from backend.main import (
    plot_spectral_profiles_interactive,
    plot_band_accuracy_curve_interactive,
)

def render_raw_spectra():
    # Helper CSS for inner cards
    st.markdown("""
        <style>
        .viz-card {
            background-color: #0F3B34; /* Slightly lighter than panel */
            border: 1px solid #134E4A;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    selected = st.session_state.get("selected_model_data")
    if selected is None:
        st.caption("Select a model to view spectra.")
        return

    X_raw = st.session_state.get("X_raw")
    band_names = st.session_state.get("band_names")

    # Render Plot inside a card div
    st.markdown('<div class="viz-card">', unsafe_allow_html=True)
    fig = plot_spectral_profiles_interactive(X_raw, band_names=band_names)
    
    # Tight layout for the side panel
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10), 
        height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_band_accuracy():
    selected = st.session_state.get("selected_model_data")
    if selected is None:
        st.caption("Select a model to view accuracy.")
        return

    model = st.session_state.get("loaded_model")
    X_prep = st.session_state.get("X_prep")
    y_data = st.session_state.get("y_data")
    band_names = st.session_state.get("band_names")

    # Render Plot inside a card div
    st.markdown('<div class="viz-card">', unsafe_allow_html=True)
    fig_curve = plot_band_accuracy_curve_interactive(model, X_prep, y_data, band_names)
    
    # Tight layout
    fig_curve.update_layout(
        margin=dict(l=10, r=10, t=30, b=10), 
        height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_curve, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)