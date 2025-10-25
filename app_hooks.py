import streamlit as st
from taste import TasteState

if "taste" not in st.session_state:
    st.session_state.taste = TasteState()
taste = st.session_state.taste
