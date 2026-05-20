import streamlit as st
from api_utils import get_api_response


def handle_api_response(response):
    if response is None:
        st.error("Keine Antwort vom Backend erhalten.")
        st.stop()

    if not isinstance(response, dict):
        st.error(f"Ungültige Backend-Antwort: {response}")
        st.stop()

    if response.get("error"):
        st.error(response["error"])
        st.stop()

    if response.get("session_id"):
        st.session_state.session_id = response["session_id"]

    st.session_state.last_response = response
    return response


def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def display_static_qa(last_response):
    static_qa = last_response.get("static_qa") or {}

    question = static_qa.get("question")
    options = static_qa.get("answer", [])

    if not question:
        st.error(f"Static QA response missing question: {last_response}")
        st.stop()

    if not isinstance(options, list):
        options = [str(options)]

    with st.chat_message("assistant"):
        st.markdown(question)

    with st.form("qa_form"):
        selected_value = None

        if options:
            selected_value = st.radio(
                "Bitte auswählen:",
                options,
                key="qa_radio"
            )

        custom_input = st.text_input(
            "Oder eigene Antwort eingeben:",
            key="custom_answer"
        )

        submitted = st.form_submit_button("Antwort senden")

    if not submitted:
        return

    final_answer = custom_input.strip() if custom_input else ""

    if not final_answer and selected_value:
        final_answer = selected_value

    if not final_answer:
        st.error("Bitte eine Antwort auswählen oder eingeben.")
        st.stop()

    st.session_state.messages.append({
        "role": "assistant",
        "content": question
    })

    st.session_state.messages.append({
        "role": "user",
        "content": final_answer
    })

    with st.spinner("Verarbeite..."):
        response = get_api_response(
            query=final_answer,
            session_id=st.session_state.session_id,
            model=st.session_state.model
        )

        response = handle_api_response(response)

        if response.get("is_done", True):
            assistant_response = response.get("response")
            if assistant_response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_response
                })

        st.rerun()


def display_chat_interface():
    required_keys = ["messages", "session_id", "model", "last_response"]

    for key in required_keys:
        if key not in st.session_state:
            raise RuntimeError(f"Missing required session state key: '{key}'")

    display_chat_history()

    last_response = st.session_state.last_response or {}

    if not last_response.get("is_done", True) and last_response.get("static_qa"):
        display_static_qa(last_response)
        return

    query = st.chat_input("Deine Anfrage:")

    if not query:
        return

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Antwort wird generiert..."):
        response = get_api_response(
            query=query,
            session_id=st.session_state.session_id,
            model=st.session_state.model
        )

        response = handle_api_response(response)

        if not response.get("is_done", True) and response.get("static_qa"):
            st.rerun()

        assistant_response = response.get("response")

        if assistant_response:
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_response
            })

            with st.chat_message("assistant"):
                st.markdown(assistant_response)