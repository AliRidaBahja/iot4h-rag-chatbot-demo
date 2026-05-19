import streamlit as st
from api_utils import get_api_response

def display_chat_interface():
    # Ensure required session keys exist
    required_keys = ["messages", "session_id", "model", "last_response"]
    for key in required_keys:
        if key not in st.session_state:
            raise RuntimeError(f"Missing required session state key: '{key}'")

    # Display current chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.last_response.get("is_done"):
        query =  st.chat_input("Deine Anfrage:")
        if query: 

            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.spinner("Antwort wird generiert..."):
                response = get_api_response(
                    query=query,
                    session_id=st.session_state.session_id,
                    model=st.session_state.model
                )

                st.session_state.session_id = response["session_id"]
                st.session_state.last_response = response
                
                if not response.get("is_done"):
                    st.rerun()
                
                st.session_state.messages.append({"role": "assistant", "content": response.get("response")})
                with st.chat_message("assistant"):
                    st.markdown(response.get("response"))
                if response.get("sources"):
                    with st.expander("Details"):
                        st.subheader("Retrieved Documents")
                        for idx, doc in enumerate(response["sources"], 1):
                            st.markdown(f"**Document {idx}**")
                            st.code(doc["page_content"])
                            st.caption(str(doc["metadata"]))



    # Show static QA if available
    elif st.session_state.last_response.get("static_qa"):
        static_qa = st.session_state.last_response.get("static_qa")
        question = static_qa.get("question")
        options = static_qa.get("answer", [])

        with st.chat_message("assistant"):
            st.markdown(question)

        with st.form("qa_form"):
            selected_value = st.radio("Bitte auswählen:", options, key="qa_radio")
            selected = options.index(selected_value)
            custom_input = st.text_input("Oder eigene Antwort eingeben:", key="custom_answer")
            submitted = st.form_submit_button("Antwort senden")
            
        if submitted:
            final_answer =  custom_input.strip() if custom_input.strip() else selected_value
            st.session_state.messages.append({"role": "assistant", "content": question})
            st.session_state.messages.append({"role": "user", "content": final_answer})
            with st.chat_message("user"):
                st.markdown(final_answer)

            with st.spinner("Verarbeite..."):
                response = get_api_response(
                    query=str(final_answer),
                    session_id=st.session_state.session_id,
                    model=st.session_state.model
                )
                st.session_state.session_id = response["session_id"]
                st.session_state.last_response = response
                if st.session_state.last_response.get("is_done"):
                    st.session_state.messages.append({"role": "assistant", "content": response.get("response")})

                st.rerun()






