"""Entrypoint."""

import streamlit as st

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

with st.sidebar:
    st.metric(label="Gas price", value=4, delta=-0.5, delta_color="inverse")
    st.metric(label="Gas price", value=4, delta=-0.5, delta_color="inverse")
    st.metric(label="Gas price", value=4, delta=-0.5, delta_color="inverse")
    st.metric(label="Gas price", value=4, delta=-0.5, delta_color="inverse")
    st.metric(label="Gas price", value=4, delta=-0.5, delta_color="inverse")
    st.metric(label="Gas price", value=4, delta=-0.5, delta_color="inverse")
    st.metric(label="Gas price", value=4, delta=-0.5, delta_color="inverse")
    st.metric(label="Gas price", value=4, delta=-0.5, delta_color="inverse")
    st.metric(label="Gas price", value=4, delta=-0.5, delta_color="inverse")
    st.metric(label="Gas price", value=4, delta=-0.5, delta_color="inverse")
    st.metric(label="Gas price", value=4, delta=-0.5, delta_color="inverse")
    st.metric(label="Gas price", value=4, delta=-0.5, delta_color="inverse")
    st.metric(label="Gas price", value=4, delta=-0.5, delta_color="inverse")
    st.metric(label="Gas price", value=4, delta=-0.5, delta_color="inverse")
    st.metric(label="Gas price", value=4, delta=-0.5, delta_color="inverse")
    st.metric(label="Gas price", value=4, delta=-0.5, delta_color="inverse")
    st.metric(label="Gas price", value=4, delta=-0.5, delta_color="inverse")
    st.metric(label="Gas price", value=4, delta=-0.5, delta_color="inverse")
    st.metric(label="Gas price", value=4, delta=-0.5, delta_color="inverse")
    st.metric(label="Gas price", value=4, delta=-0.5, delta_color="inverse")
    st.metric(label="Gas price", value=4, delta=-0.5, delta_color="inverse")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
        with st.popover("Open popover"):
            st.markdown("Hello World 👋")
            name = st.text_input("What's your name?")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
