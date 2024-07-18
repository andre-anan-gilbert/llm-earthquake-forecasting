"""App entrypoint."""

import logging

import pandas as pd
import streamlit as st

from api import count_earthquakes, forecast_earthquakes, get_recent_earthquakes

logging.basicConfig(
    filename="app.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

with st.sidebar:
    df = get_recent_earthquakes(limit=10)
    df.time = pd.to_datetime(df.time)
    df.time = df.time.dt.strftime("%Y-%m-%d %H:%M:%S")
    for _, row in df.iterrows():
        with st.container(border=True):
            st.subheader(row.place)
            st.text(f"Date: {row.time}")
            st.text(f"Magnitude: {row.mag}")
            st.text(f"Depth: {row.depth} km")

col1, col2, col3, col4 = st.columns(4)
with col1.container(border=True):
    response = count_earthquakes()
    num_earthquakes_past_month = response["count"]
    st.metric(
        label="Earthquakes",
        value=num_earthquakes_past_month,
        delta="Last 30 days",
        delta_color="off",
        help="Estimated Fatalities: 0, Estimated Losses (USD): < $1 million",
    )

with col2.container(border=True):
    response = count_earthquakes(alert_level="yellow")
    num_local_earthquakes_past_month = response["count"]
    st.metric(
        label="Local/Regional Earthquakes",
        value=num_local_earthquakes_past_month,
        delta="Last 30 days",
        delta_color="off",
        help="Estimated Fatalities: 1 - 99, Estimated Losses (USD): $1 million - $100 million",
    )

with col3.container(border=True):
    response = count_earthquakes(alert_level="orange")
    num_national_earthquakes_past_month = response["count"]
    st.metric(
        label="National Earthquakes",
        value=num_national_earthquakes_past_month,
        delta="Last 30 days",
        delta_color="off",
        help="Estimated Fatalities: 100 - 999, Estimated Losses (USD): $100 million - $1 billion",
    )

with col4.container(border=True):
    response = count_earthquakes(alert_level="red")
    num_international_earthquakes_past_month = response["count"]
    st.metric(
        label="International Earthquakes",
        value=num_international_earthquakes_past_month,
        delta="Last 30 days",
        delta_color="off",
        help="Estimated Fatalities: 1,000+, Estimated Losses (USD): $1 billion+",
    )

df = forecast_earthquakes()
st.map(
    df,
    latitude="Latitude",
    longitude="Longitude",
    size=300,
    color="#90ee90",
    use_container_width=True,
)
st.dataframe(df, hide_index=True, use_container_width=True)
