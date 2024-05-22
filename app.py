"""Entrypoint."""

from datetime import datetime, timedelta
from typing import Any
from urllib import parse

import pandas as pd
import requests
import streamlit as st
from pydantic import BaseModel, Field

from language_models.agents.react import ReActAgent
from language_models.models.llm import ChatMessage, ChatMessageRole, OpenAILanguageModel
from language_models.proxy_client import BTPProxyClient
from language_models.settings import settings
from language_models.tools.current_date import current_date_tool
from language_models.tools.earthquake import earthquake_tools

st.set_page_config(layout="wide", initial_sidebar_state="expanded")


@st.cache_data
def get_recent_earthquakes(
    start_time: datetime = (datetime.now() - timedelta(days=30)).date(),
    end_time: datetime = datetime.now().date(),
    limit: int = 20000,
    min_depth: int = -100,
    max_depth: int = 1000,
    min_magnitude: int | None = None,
    max_magnitude: int | None = None,
    alert_level: str | None = None,
) -> pd.DataFrame:
    """Returns recent earthquakes."""
    params = {
        "format": "csv",
        "starttime": start_time,
        "endtime": end_time,
        "limit": limit,
        "mindepth": min_depth,
        "maxdepth": max_depth,
        "eventtype": "earthquake",
    }
    if min_magnitude is not None:
        params["minmagnitude"] = min_magnitude
    if max_magnitude is not None:
        params["maxmagnitude"] = max_magnitude
    if alert_level is not None:
        params["alertlevel"] = alert_level
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?" + parse.urlencode(params)
    return pd.read_csv(url)


@st.cache_data
def count_earthquakes(
    start_time: datetime = (datetime.now() - timedelta(days=30)).date(),
    end_time: datetime = datetime.now().date(),
    limit: int = 20000,
    min_depth: int = -100,
    max_depth: int = 1000,
    min_magnitude: int | None = None,
    max_magnitude: int | None = None,
    alert_level: str | None = None,
) -> int:
    params = {
        "format": "geojson",
        "starttime": start_time,
        "endtime": end_time,
        "limit": limit,
        "mindepth": min_depth,
        "maxdepth": max_depth,
        "minmagnitude": min_magnitude,
        "maxmagnitude": max_magnitude,
        "alertlevel": alert_level,
        "eventtype": "earthquake",
    }
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
    df = get_recent_earthquakes(limit=20)
    for _, row in df.iterrows():
        with st.container(border=True):
            st.subheader(row.place)
            st.text(f"Magnitude: {row.mag}")
            st.text(f"Depth: {row.depth}")
            st.text(f"Date: {row.time}")


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
        max_tokens=512,
        temperature=0.0,
    )

    system_prompt = (
        "You are an United States Geological Survey expert who can answer questions regarding earthquakes"
        + " and can run forecasts."
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


def display_widget(messenger, tool: dict[str, Any] | None) -> None:
    if tool is None:
        return
    elif tool["name"] == "Count Earthquakes":
        count = count_earthquakes(**tool["args"])["count"]
        messenger.metric(label="Number of Earthquakes", value=count)
    elif tool["name"] == "Query Earthquakes":
        data = get_recent_earthquakes(**tool["args"])
        messenger.map(
            data,
            latitude="latitude",
            longitude="longitude",
            size=1000,
            color="#90ee90",
        )


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
chat_history = []
for message in st.session_state.messages:
    prev_steps = []
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            display_widget(st, message["last_tool"])
            popover = st.popover("View Reasoning", use_container_width=True)
            for progress in message["chain_of_thought"]:
                if progress["step"] == "final_answer":
                    popover.markdown(
                        f":green-background[Final Answer] {progress['content']['content']}"
                    )
                elif progress["step"] == "tool":
                    tool_name = progress["content"]["name"]
                    tool_input = progress["content"]["args"]
                    tool_response = progress["content"]["response"]
                    popover.markdown(f":orange-background[Tool] {tool_name}")
                    popover.markdown(f":orange-background[Tool Input] {tool_input}")
                    popover.markdown(
                        f":orange-background[Tool Response] {tool_response}"
                    )
                    prev_steps.append(f"Tool: {tool_name}")
                    prev_steps.append(f"Tool Input: {tool_input}")
                    prev_steps.append(f"Tool Response: {tool_response}")
                else:
                    thought = progress["content"]
                    popover.markdown(f":blue-background[Thought] {thought}")
                    prev_steps.append(f"Thought: {thought}")

    if prev_steps:
        chat_history[-1].content += "\n\nThis was your previous work:\n\n" + "\n".join(
            prev_steps
        )

    chat_history.append(
        ChatMessage(
            role=(
                ChatMessageRole.USER
                if message["role"] == "user"
                else ChatMessageRole.ASSISTANT
            ),
            content=(message["content"]),
        )
    )

agent = get_agent()
agent.chat_messages = [agent.chat_messages[0]] + chat_history

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
            last_tool = stream["content"].last_tool

            # Add assistant response to chat history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_answer,
                    "chain_of_thought": chain_of_thought,
                    "last_tool": last_tool,
                }
            )

    st.rerun()
