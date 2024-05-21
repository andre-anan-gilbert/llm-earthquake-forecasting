"""Entrypoint."""

from datetime import datetime, timedelta
from typing import Any
from urllib import parse

import pandas as pd
import requests
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
    """Returns recent earthquakes."""
    params = {
        "format": "csv",
        "eventtype": "earthquake",
        "limit": 30,
        "starttime": (datetime.now() - timedelta(days=30)).date(),
        "endtime": (datetime.now().date()),
    }
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?" + parse.urlencode(params)
    return pd.read_csv(url)


@st.cache_data
def count_earthquakes(alert_level: str | None = None) -> int:
    start_time = (datetime.now() - timedelta(days=30)).date()
    end_time = datetime.now().date()
    params = {
        "format": "geojson",
        "starttime": start_time,
        "endtime": end_time,
        "limit": 20000,
        "eventtype": "earthquake",
    }
    if alert_level is not None:
        params["alertlevel"] = alert_level
    return requests.get(
        "https://earthquake.usgs.gov/fdsnws/event/1/count",
        params=params,
        timeout=None,
    ).json()


# Display metrics and recent earthquakes in sidebar
with st.sidebar:
    st.title("Earthquake Forecasting")
    col1, col2 = st.columns(2)
    with col1:
        response = count_earthquakes()
        num_earthquakes_past_month = response["count"]
        st.metric(
            label="Earthquakes",
            value=num_earthquakes_past_month,
            delta="Last 30 days",
            delta_color="off",
        )
        response = count_earthquakes(alert_level="orange")
        num_national_earthquakes_past_month = response["count"]
        st.metric(
            label="National Earthquakes",
            value=num_national_earthquakes_past_month,
            delta="Last 30 days",
            delta_color="off",
        )
    with col2:
        response = count_earthquakes(alert_level="yellow")
        num_local_earthquakes_past_month = response["count"]
        st.metric(
            label="Local/Regional Earthquakes",
            value=num_local_earthquakes_past_month,
            delta="Last 30 days",
            delta_color="off",
        )
        response = count_earthquakes(alert_level="red")
        num_international_earthquakes_past_month = response["count"]
        st.metric(
            label="International Earthquakes",
            value=num_international_earthquakes_past_month,
            delta="Last 30 days",
            delta_color="off",
        )

    st.divider()
    st.header("Recent Earthquakes")
    df = get_recent_earthquakes()
    for _, data in df.iterrows():
        with st.container(border=True):
            st.subheader(data.place)
            st.text(f"Magnitude: {data.mag}")
            st.text(f"Depth: {data.depth}")
            st.text(f"Date: {data.time}")


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


def render_chain_of_thought_step(cot_step: dict[str, Any]) -> None:
    if cot_step["step"] == "final_answer":
        st.markdown(f":green-background[Final Answer] {cot_step['content']['content']}")
    elif cot_step["step"] == "tool":
        st.markdown(f":orange-background[Tool] {cot_step['content']['name']}")
        st.markdown(f":orange-background[Tool Input] {cot_step['content']['args']}")
        st.markdown(
            f":orange-background[Tool Response] {cot_step['content']['response']}"
        )
    else:
        st.markdown(f":blue-background[Thought] {cot_step['content']}")


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            with st.popover("View Reasoning", use_container_width=True):
                for progress in message["chain_of_thought"]:
                    render_chain_of_thought_step(progress)

# React to user input
if prompt := st.chat_input("Message Earthquake Agent"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    assistant = st.chat_message("assistant")
    status = assistant.status(label="Running...")
    for stream in agent.invoke({"prompt": prompt}):
        if stream["step"] == "thought":
            status.update(label=stream["content"])
        elif stream["step"] == "tool":
            status.update(label=f"Using Tool: {stream['content']}")
        else:
            # Display assistant response in chat message container
            status.update(label="Done!", state="complete")
            final_answer = stream["content"].final_answer["content"]
            chain_of_thought = stream["content"].chain_of_thought
            assistant.markdown(final_answer)
            with assistant.popover("View Reasoning", use_container_width=True):
                for progress in chain_of_thought:
                    render_chain_of_thought_step(progress)

            # Add assistant response to chat history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_answer,
                    "chain_of_thought": chain_of_thought,
                }
            )
