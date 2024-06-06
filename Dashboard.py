"""App entrypoint."""

import plotly.express as px
import streamlit as st

from api import count_earthquakes, get_recent_earthquakes

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

col1, col2, col3, col4 = st.columns(4)
with col1.container(border=True):
    response = count_earthquakes()
    num_earthquakes_past_month = response["count"]
    st.metric(
        label="Earthquakes",
        value=num_earthquakes_past_month,
        delta="Last 30 days",
        delta_color="off",
    )

with col2.container(border=True):
    response = count_earthquakes(alert_level="orange")
    num_national_earthquakes_past_month = response["count"]
    st.metric(
        label="National Earthquakes",
        value=num_national_earthquakes_past_month,
        delta="Last 30 days",
        delta_color="off",
    )
with col3.container(border=True):
    response = count_earthquakes(alert_level="yellow")
    num_local_earthquakes_past_month = response["count"]
    st.metric(
        label="Local/Regional Earthquakes",
        value=num_local_earthquakes_past_month,
        delta="Last 30 days",
        delta_color="off",
    )

with col4.container(border=True):
    response = count_earthquakes(alert_level="red")
    num_international_earthquakes_past_month = response["count"]
    st.metric(
        label="International Earthquakes",
        value=num_international_earthquakes_past_month,
        delta="Last 30 days",
        delta_color="off",
    )

st.header("Recent Earthquakes")
df = get_recent_earthquakes()
st.dataframe(df)
