"""AI agent."""

import os
from datetime import datetime, timedelta
from typing import Any

import catboost as cb
import pandas as pd
import pytz
import requests
import streamlit as st
from pydantic import BaseModel, Field

from api import count_earthquakes, get_recent_earthquakes
from language_models.agents.react import ReActAgent
from language_models.models.llm import OpenAILanguageModel
from language_models.proxy_client import BTPProxyClient
from language_models.settings import settings
from language_models.tools.tool import Tool

_START_LAG = 3
_END_LAG = 12


def get_regions() -> list[str]:
    df = get_recent_earthquakes()
    df["region"] = df.place.str.split(", ", expand=True)[1]
    df.region = df.region.fillna(df.place)
    df.region = df.region.replace("CA", "California")
    df.region = df.region.replace("B.C.", "Baja California")
    return set(
        [
            "California",
            "Alaska",
            "Nevada",
            "Hawaii",
            "Washington",
            "Utah",
            "Montana",
            "Puerto Rico",
            "Indonesia",
            "Chile",
            "Baja California",
            "Oklahoma",
            "Japan",
            "Greece",
            "Papua New Guinea",
            "Philippines",
            "Mexico",
            "Italy",
            "Russia",
            "Idaho",
            "Aleutian Islands",
            "Tonga",
            "Oregon",
            "Wyoming",
            "Turkey",
        ]
    ) & set(df.region.unique())


@st.cache_resource
def load_model() -> cb.CatBoostRegressor:
    path = os.path.join(os.path.dirname(__file__), "./ml/model_depth_8")
    model = cb.CatBoostRegressor(cat_features=["region"])
    return model.load_model(path)


def preprocess_data(df: pd.DataFrame, region: str) -> pd.DataFrame:
    df = df.copy()

    df["region"] = df.place.str.split(", ", expand=True)[1]
    df.region = df.region.fillna(df.place)
    df.region = df.region.replace("CA", "California")
    df.region = df.region.replace("B.C.", "Baja California")

    df.time = pd.to_datetime(df.time)
    df = df.sort_values("time")
    df = df.set_index("time")

    df = df.loc[df.region == region]
    df = df[["depth", "mag", "region", "latitude", "longitude"]]

    df = df.groupby("region").resample("d").mean()
    df = df.reset_index()

    df = df.set_index("time")

    start_date = df.index.min()
    end_date = pd.Timestamp(datetime.today().date(), tz=pytz.UTC)
    date_range = pd.date_range(start=start_date, end=end_date, freq="d")
    df = df.reindex(date_range)

    df.mag = df.mag.ffill()
    df.depth = df.depth.ffill()
    df.latitude = df.latitude.ffill()
    df.longitude = df.longitude.ffill()

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    start_date = df.index.min()
    end_date = pd.Timestamp((datetime.now() + timedelta(days=3)).date(), tz=pytz.UTC)
    date_range = pd.date_range(start=start_date, end=end_date, freq="d")
    df = df.reindex(date_range)

    df.region = df.region.ffill()

    df["day"] = df.index.day
    df["dayofweek"] = df.index.dayofweek
    df["dayofyear"] = df.index.dayofyear

    for i in range(_START_LAG, _END_LAG + 1):
        df[f"mag_lag_{i}"] = df.groupby("region").mag.shift(i)

    df[f"mag_rolling_mean_{_START_LAG}"] = df.groupby("region").mag.transform(
        lambda x: x.rolling(window=_START_LAG).mean()
    )
    df[f"mag_rolling_std_{_START_LAG}"] = df.groupby("region").mag.transform(
        lambda x: x.rolling(window=_START_LAG).std()
    )
    df[f"depth_rolling_mean_{_START_LAG}"] = df.groupby("region").depth.transform(
        lambda x: x.rolling(window=_START_LAG).mean()
    )
    df[f"depth_rolling_std_{_START_LAG}"] = df.groupby("region").depth.transform(
        lambda x: x.rolling(window=_START_LAG).std()
    )
    df[f"latitude_rolling_mean_{_START_LAG}"] = df.groupby("region").latitude.transform(
        lambda x: x.rolling(window=_START_LAG).mean()
    )
    df[f"latitude_rolling_std_{_START_LAG}"] = df.groupby("region").latitude.transform(
        lambda x: x.rolling(window=_START_LAG).std()
    )
    df[f"longitude_rolling_mean_{_START_LAG}"] = df.groupby(
        "region"
    ).longitude.transform(lambda x: x.rolling(window=_START_LAG).mean())
    df[f"longitude_rolling_std_{_START_LAG}"] = df.groupby(
        "region"
    ).longitude.transform(lambda x: x.rolling(window=_START_LAG).std())

    df[f"mag_rolling_mean_{_END_LAG}"] = df.groupby("region").mag.transform(
        lambda x: x.rolling(window=_END_LAG).mean()
    )
    df[f"mag_rolling_std_{_END_LAG}"] = df.groupby("region").mag.transform(
        lambda x: x.rolling(window=_END_LAG).std()
    )
    df[f"depth_rolling_mean_{_END_LAG}"] = df.groupby("region").depth.transform(
        lambda x: x.rolling(window=_END_LAG).mean()
    )
    df[f"depth_rolling_std_{_END_LAG}"] = df.groupby("region").depth.transform(
        lambda x: x.rolling(window=_END_LAG).std()
    )
    df[f"latitude_rolling_mean_{_END_LAG}"] = df.groupby("region").latitude.transform(
        lambda x: x.rolling(window=_END_LAG).mean()
    )
    df[f"latitude_rolling_std_{_END_LAG}"] = df.groupby("region").latitude.transform(
        lambda x: x.rolling(window=_END_LAG).std()
    )
    df[f"longitude_rolling_mean_{_END_LAG}"] = df.groupby("region").longitude.transform(
        lambda x: x.rolling(window=_END_LAG).mean()
    )
    df[f"longitude_rolling_std_{_END_LAG}"] = df.groupby("region").longitude.transform(
        lambda x: x.rolling(window=_END_LAG).std()
    )

    return df


def get_forecast(region: str) -> pd.DataFrame:
    model = load_model()
    df = get_recent_earthquakes()
    df = preprocess_data(df, region)
    df = create_features(df)
    features = [
        "day",
        "dayofweek",
        "dayofyear",
        f"mag_rolling_mean_{_START_LAG}",
        f"mag_rolling_std_{_START_LAG}",
        f"depth_rolling_mean_{_START_LAG}",
        f"depth_rolling_std_{_START_LAG}",
        f"latitude_rolling_mean_{_START_LAG}",
        f"latitude_rolling_std_{_START_LAG}",
        f"longitude_rolling_mean_{_START_LAG}",
        f"longitude_rolling_std_{_START_LAG}",
        f"mag_rolling_mean_{_END_LAG}",
        f"mag_rolling_std_{_END_LAG}",
        f"depth_rolling_mean_{_END_LAG}",
        f"depth_rolling_std_{_END_LAG}",
        f"latitude_rolling_mean_{_END_LAG}",
        f"latitude_rolling_std_{_END_LAG}",
        f"longitude_rolling_mean_{_END_LAG}",
        f"longitude_rolling_std_{_END_LAG}",
    ] + [f"mag_lag_{i}" for i in range(_START_LAG, _END_LAG + 1)]
    cat_features = ["region"]
    target = "mag"
    forecast = model.predict(df[features + cat_features])
    return pd.DataFrame({"magnitude": df[target], "forecast": forecast}, index=df.index)


class Forecast(BaseModel):
    region: str = Field(
        description="A valid region that the model can perform forecasts for."
    )


def forecast_formatted(region: str) -> dict[str, list]:
    df_forecast = get_forecast(region)
    df_forecast = df_forecast.rename(columns={"magnitude": "actual magnitude"})
    return {"forecast": df_forecast.to_dict(orient="records")}


class USGeopoliticalSurveyEarthquakeAPI(BaseModel):
    """Class that implements the API interface."""

    start_time: str = Field(
        None,
        description=(
            "Limit to events on or after the specified start time. NOTE: All times use ISO8601 Date/Time format."
            + " Unless a timezone is specified, UTC is assumed."
        ),
    )
    end_time: str = Field(
        None,
        description=(
            "Limit to events on or before the specified end time. NOTE: All times use ISO8601 Date/Time format."
            + " Unless a timezone is specified, UTC is assumed."
        ),
    )
    limit: int = Field(
        20000,
        description=(
            "Limit the results to the specified number of events. NOTE: The service limits queries to 20000,"
            + " and any that exceed this limit will generate a HTTP response code 400 Bad Request."
        ),
    )
    min_depth: int = Field(
        -100,
        description="Limit to events with depth more than the specified minimum.",
    )
    max_depth: int = Field(
        1000,
        description="Limit to events with depth less than the specified maximum.",
    )
    min_magnitude: int = Field(
        None,
        description="Limit to events with a magnitude larger than the specified minimum.",
    )
    max_magnitude: int = Field(
        None,
        description="Limit to events with a magnitude smaller than the specified maximum.",
    )
    alert_level: str = Field(
        None,
        description=(
            "Limit to events with a specific PAGER alert level."
            + " The allowed values are: alert_level=green Limit to events with PAGER"
            + ' alert level "green". alert_level=yellow Limit to events with PAGER alert level "yellow".'
            + ' alert_level=orange Limit to events with PAGER alert level "orange".'
            + ' alert_level=red Limit to events with PAGER alert level "red".'
        ),
    )
    eventtype: str = Field(
        None,
        description="Limit to events of a specific type. NOTE: “earthquake” will filter non-earthquake events.",
    )


@st.cache_data
def query_earthquakes(
    start_time: datetime = (datetime.now() - timedelta(days=30)).date(),
    end_time: datetime = datetime.now().date(),
    limit: int = 20000,
    min_depth: int = -100,
    max_depth: int = 1000,
    min_magnitude: int | None = None,
    max_magnitude: int | None = None,
    alert_level: str | None = None,
) -> Any:
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
    response = requests.get(
        "https://earthquake.usgs.gov/fdsnws/event/1/query",
        params=params,
        timeout=None,
    )
    return response.json()


def current_date() -> datetime:
    return datetime.now()


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
        max_tokens=1000,
        temperature=0.0,
    )

    system_prompt = (
        "You are an United States Geological Survey expert who can answer questions regarding earthquakes"
        + " and can run forecasts."
    )

    class Output(BaseModel):
        content: str = Field(description="The final answer.")

    tools = [
        Tool(
            func=current_date,
            name="Current Date",
            description="Use this tool to access the current local date and time.",
        ),
        Tool(
            func=query_earthquakes,
            name="Query Earthquakes",
            description="Use this tool to search recent earthquakes.",
            args_schema=USGeopoliticalSurveyEarthquakeAPI,
        ),
        Tool(
            func=count_earthquakes,
            name="Count Earthquakes",
            description="Use this tool to count and aggregate recent earthquakes.",
            args_schema=USGeopoliticalSurveyEarthquakeAPI,
        ),
        Tool(
            func=get_regions,
            name="Find Regions",
            description="Use this tool to access the regions that can be used for forecasting.",
        ),
        Tool(
            func=forecast_formatted,
            name="Forecast Earthquakes",
            description="Use this tool to forecast earthquakes in a specified region.",
            args_schema=Forecast,
        ),
    ]

    return ReActAgent.create(
        llm=llm,
        system_prompt=system_prompt,
        task_prompt="{prompt}",
        task_prompt_variables=["prompt"],
        tools=tools,
        output_format=Output,
        iterations=10,
    )
