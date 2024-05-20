"""Entrypoint."""

from datetime import datetime, timedelta
from urllib import parse

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field

from language_models.agents.react import ReActAgent
from language_models.models.llm import OpenAILanguageModel
from language_models.proxy_client import BTPProxyClient
from language_models.settings import settings
from language_models.tools.current_date import current_date_tool
from language_models.tools.earthquake import earthquake_tools

st.set_page_config(layout="wide", initial_sidebar_state="expanded")


@st.cache_data
def get_recent_earthquakes() -> pd.DataFrame:
    """Returns 100 recent earthquakes."""
    params = {
        "format": "csv",
        "eventtype": "earthquake",
        "limit": 100,
        "starttime": (datetime.now() - timedelta(days=30)).date(),
        "endtime": (datetime.now().date()),
    }
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?" + parse.urlencode(params)
    return pd.read_csv(url)


# Display 100 recent earthquakes in sidebar
with st.sidebar:
    df = get_recent_earthquakes()
    for _, data in df.iterrows():
        st.metric(
            label=data.place,
            value=data.mag,
            delta="Magnitude",
            delta_color="off",
        )


def get_agent() -> ReActAgent:
    """Returns an agent using ReAct prompting."""
    proxy_client = BTPProxyClient(
        client_id=settings.CLIENT_ID,
        client_secret=settings.CLIENT_SECRET,
        auth_url=settings.AUTH_URL,
        api_base=settings.API_BASE,
    )

    llm = OpenAILanguageModel(
        proxy_client=proxy_client,
        model="gpt-4",
        max_tokens=1024,
        temperature=0.0,
    )

    system_prompt = (
        "You are an United States Geological Survey expert who can answer questions regarding earthquakes"
        + " and can run forecasts. Use the current date tool to access the local date"
        + " and time before using other tools.Take the following question and answer it as accurately as possible."
    )

    class Output(BaseModel):
        content: str = Field(description="The final answer.")

    return ReActAgent.create(
        llm=llm,
        system_prompt=system_prompt,
        task_prompt="{prompt}",
        task_prompt_variables=["prompt"],
        tools=[current_date_tool] + earthquake_tools,
        output_format=Output,
        iterations=10,
    )


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

agent = get_agent()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            with st.popover("View reasoning"):
                st.markdown("Hello World 👋")

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = agent.invoke({"prompt": prompt})
    final_answer = response.final_answer["content"]
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(final_answer)
        with st.popover("View reasoning"):
            st.markdown("Hello World 👋")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
