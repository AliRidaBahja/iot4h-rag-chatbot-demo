import streamlit as st
import uuid
def display_sidebar():
    # Model selection
    model_options = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]

    st.sidebar.selectbox("Select Model", options=model_options, key="model")

    # Session ID input or generation
    if "session_id" not in st.session_state or not st.session_state.session_id:
        st.session_state.session_id = str(uuid.uuid4())

    # Show input field (but don't bind it directly to session_id to avoid overwrite issue)
    session_input = st.sidebar.text_input("Session ID", value=st.session_state.session_id)

    # If user manually changes the session ID
    if session_input != st.session_state.session_id:
        st.session_state.session_id = session_input