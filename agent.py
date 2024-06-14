"""AI agent."""

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests
from pydantic import BaseModel, Field

from api import count_earthquakes, get_forecast, get_regions
from language_models.agents.react import ReActAgent
from language_models.models.llm import OpenAILanguageModel
from language_models.proxy_client import BTPProxyClient
from language_models.settings import settings
from language_models.tools.tool import Tool


class Forecast(BaseModel):
    region: str = Field(description="A valid region that the model can perform forecasts for.")


def forecast_earthquakes(region: str) -> dict[str, list]:
    df = get_forecast(region)
    today = pd.Timestamp.now()
    df = df.loc[df.Date > today]
    df = df[["Date", "Magnitude Forecast", "Depth Forecast"]]
    return {"forecast": df.to_dict(orient="records")}


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
        temperature=0.2,
    )

    system_prompt = (
        "You are an United States Geological Survey expert who can answer questions regarding earthquakes"
        + " and can run forecasts. Before you use the Forecast Earthquakes tool, always check which "
        + "regions are available using Find Regions first."
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
            description="Use this tool to access the available regions that can be used for forecasting.",
        ),
        Tool(
            func=forecast_earthquakes,
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
        iterations=20,
    )
