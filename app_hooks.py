import streamlit as st
from taste import TasteState

def get_taste():
    # Always read from session state; recreate if missing
    if "taste" not in st.session_state or not isinstance(st.session_state.get("taste"), TasteState):
        st.session_state["taste"] = TasteState()
    return st.session_state["taste"]
