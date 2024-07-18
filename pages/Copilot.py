"""Entrypoint."""

from typing import Any

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agent import get_agent, get_forecast
from api import count_earthquakes, get_recent_earthquakes
from language_models.models.llm import ChatMessage, ChatMessageRole

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Display metrics and recent earthquakes in sidebar
with st.sidebar:
    with st.container(border=True):
        response = count_earthquakes()
        num_earthquakes_past_month = response["count"]
        st.metric(
            label="Earthquakes",
            value=num_earthquakes_past_month,
            delta="Last 30 days",
            delta_color="off",
            help="Estimated Fatalities: 0, Estimated Losses (USD): < $1 million",
        )

    with st.container(border=True):
        response = count_earthquakes(alert_level="yellow")
        num_local_earthquakes_past_month = response["count"]
        st.metric(
            label="Local/Regional Earthquakes",
            value=num_local_earthquakes_past_month,
            delta="Last 30 days",
            delta_color="off",
            help="Estimated Fatalities: 1 - 99, Estimated Losses (USD): $1 million - $100 million",
        )

    with st.container(border=True):
        response = count_earthquakes(alert_level="orange")
        num_national_earthquakes_past_month = response["count"]
        st.metric(
            label="National Earthquakes",
            value=num_national_earthquakes_past_month,
            delta="Last 30 days",
            delta_color="off",
            help="Estimated Fatalities: 100 - 999, Estimated Losses (USD): $100 million - $1 billion",
        )

    with st.container(border=True):
        response = count_earthquakes(alert_level="red")
        num_international_earthquakes_past_month = response["count"]
        st.metric(
            label="International Earthquakes",
            value=num_international_earthquakes_past_month,
            delta="Last 30 days",
            delta_color="off",
            help="Estimated Fatalities: 1,000+, Estimated Losses (USD): $1 billion+",
        )


def display_widget(messenger, tool: dict[str, Any] | None) -> None:
    if tool is None:
        return
    elif tool["name"] == "Count Earthquakes":
        count = count_earthquakes(**tool["args"])["count"]
        messenger.metric(label="Number of Earthquakes", value=count)
    elif tool["name"] == "Query Earthquakes":
        data = get_recent_earthquakes(**tool["args"])
        tab1, tab2 = messenger.tabs(["Map", "Data"])
        with tab1:
            messenger.map(
                data,
                latitude="latitude",
                longitude="longitude",
                size=300,
                color="#90ee90",
                use_container_width=True,
            )
        with tab2:
            messenger.dataframe(data)
    elif tool["name"] == "Forecast Earthquakes":
        df_forecast = get_forecast(**tool["args"])
        tab1, tab2 = messenger.tabs(["Magnitude", "Depth"])
        with tab1:
            fig = px.line(
                df_forecast,
                x="Date",
                y=["Magnitude", "Magnitude Forecast"],
                markers=True,
            )
            fig.update_layout(
                title=f"Magnitude Forecast for {tool['args']['region']}",
                yaxis_title="Magnitude",
                legend_title_text="Forecast",
            )
            fig.add_traces(
                [
                    go.Scatter(
                        x=df_forecast["Date"],
                        y=df_forecast["Upper 90 Magnitude Forecast"],
                        mode="lines",
                        line_color="rgba(0,0,0,0)",
                        showlegend=False,
                    ),
                    go.Scatter(
                        x=df_forecast["Date"],
                        y=df_forecast["Lower 90 Magnitude Forecast"],
                        mode="lines",
                        line_color="rgba(0,0,0,0)",
                        name="90% Confidence interval",
                        fill="tonexty",
                        fillcolor="rgba(231,107,243,0.2)",
                    ),
                    go.Scatter(
                        x=df_forecast["Date"],
                        y=df_forecast["Upper 50 Magnitude Forecast"],
                        mode="lines",
                        line_color="rgba(0,0,0,0)",
                        showlegend=False,
                    ),
                    go.Scatter(
                        x=df_forecast["Date"],
                        y=df_forecast["Lower 50 Magnitude Forecast"],
                        mode="lines",
                        line_color="rgba(0,0,0,0)",
                        name="50% Confidence interval",
                        fill="tonexty",
                        fillcolor="rgba(0,176,246,0.2)",
                    ),
                ]
            )
            messenger.plotly_chart(fig, use_container_width=True)
        with tab2:
            fig = px.line(
                df_forecast,
                x="Date",
                y=["Depth", "Depth Forecast"],
                markers=True,
            )
            fig.update_layout(
                title=f"Depth Forecast for {tool['args']['region']}",
                yaxis_title="Depth",
                legend_title_text="Forecast",
            )
            fig.add_traces(
                [
                    go.Scatter(
                        x=df_forecast["Date"],
                        y=df_forecast["Upper 90 Depth Forecast"],
                        mode="lines",
                        line_color="rgba(0,0,0,0)",
                        showlegend=False,
                    ),
                    go.Scatter(
                        x=df_forecast["Date"],
                        y=df_forecast["Lower 90 Depth Forecast"],
                        mode="lines",
                        line_color="rgba(0,0,0,0)",
                        name="90% Confidence interval",
                        fill="tonexty",
                        fillcolor="rgba(231,107,243,0.2)",
                    ),
                    go.Scatter(
                        x=df_forecast["Date"],
                        y=df_forecast["Upper 50 Depth Forecast"],
                        mode="lines",
                        line_color="rgba(0,0,0,0)",
                        showlegend=False,
                    ),
                    go.Scatter(
                        x=df_forecast["Date"],
                        y=df_forecast["Lower 50 Depth Forecast"],
                        mode="lines",
                        line_color="rgba(0,0,0,0)",
                        name="50% Confidence interval",
                        fill="tonexty",
                        fillcolor="rgba(0,176,246,0.2)",
                    ),
                ]
            )
            messenger.plotly_chart(fig, use_container_width=True)


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
                    popover.markdown(f":green-background[Final Answer] {progress['content']['content']}")
                elif progress["step"] == "tool":
                    tool_name = progress["content"]["name"]
                    tool_input = progress["content"]["args"]
                    tool_response = progress["content"]["response"]
                    popover.markdown(f":orange-background[Tool] {tool_name}")
                    popover.markdown(f":orange-background[Tool Input] {tool_input}")
                    popover.markdown(f":orange-background[Tool Response] {tool_response}")
                    prev_steps.append(f"Tool: {tool_name}")
                    prev_steps.append(f"Tool Input: {tool_input}")
                    prev_steps.append(f"Tool Response: {tool_response}")
                else:
                    thought = progress["content"]
                    popover.markdown(f":blue-background[Thought] {thought}")
                    prev_steps.append(f"Thought: {thought}")

    if prev_steps:
        chat_history[-1].content += "\n\nThis was your previous work:\n\n" + "\n".join(prev_steps)

    chat_history.append(
        ChatMessage(
            role=(ChatMessageRole.USER if message["role"] == "user" else ChatMessageRole.ASSISTANT),
            content=str(message["content"]),
        )
    )

agent = get_agent()
agent.chat_messages = [agent.chat_messages[0]] + chat_history

# React to user input
if prompt := st.chat_input("Ask Earthquake Copilot"):
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
