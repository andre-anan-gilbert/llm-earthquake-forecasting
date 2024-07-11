"""API calls."""

import os
from datetime import datetime, timedelta
from urllib import parse

import catboost as cb
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import norm

_LAGS = 7
_MEDIAN_LATITUDE = {
    "Alaska": 61.1224,
    "Aleutian Islands": 51.7672,
    "Baja California": 32.36241665,
    "California": 36.6798333,
    "Chile": -32.443,
    "Greece": 38.07,
    "Hawaii": 19.3948326,
    "Idaho": 44.3036667,
    "Indonesia": -1.9559,
    "Italy": 44.204,
    "Japan": 36.306,
    "Mexico": 16.629,
    "Montana": 44.7548333,
    "Nevada": 38.158,
    "Oklahoma": 36.1025,
    "Oregon": 43.88341665,
    "Papua New Guinea": -5.604,
    "Philippines": 9.242,
    "Puerto Rico": 17.9971,
    "Russia": 50.895,
    "Tonga": -18.9217,
    "Turkey": 39.117,
    "Utah": 39.4306667,
    "Washington": 46.5871667,
    "Wyoming": 4,
}


_MEDIAN_LONGITUDE = {
    "Alaska": -151.1221,
    "Aleutian Islands": 178.314,
    "Baja California": -115.57125,
    "California": -118.8568333,
    "Chile": -71.303,
    "Greece": 22.52,
    "Hawaii": -155.2838333,
    "Idaho": -114.5975,
    "Indonesia": 122.566,
    "Italy": 10.717,
    "Japan": 141.044,
    "Mexico": -98.208,
    "Montana": -111.0218333,
    "Nevada": -117.8719,
    "Oklahoma": -97.5726,
    "Oregon": -121.93625,
    "Papua New Guinea": 151.269,
    "Philippines": 125.822,
    "Puerto Rico": -66.8571,
    "Russia": 150.9206,
    "Tonga": -174.5247,
    "Turkey": 28.975,
    "Utah": -111.4066667,
    "Washington": -122.1856667,
    "Wyoming": -110.7056667,
}


def get_regions() -> list[str]:
    df = get_recent_earthquakes()
    df["region"] = df.place.str.split(", ", expand=True)[1]
    df.region = df.region.fillna(df.place)
    df.region = df.region.replace({"CA": "California", "B.C.": "Baja California"})
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


@st.cache_resource
def load_model() -> cb.CatBoostRegressor:
    path = os.path.join(os.path.dirname(__file__), "./ml/forecasting_model")
    model = cb.CatBoostRegressor(cat_features=["region"])
    return model.load_model(path)


def reindex(group, delta):
    start_date = group.index.min()
    end_date = pd.Timestamp((datetime.now() + timedelta(days=delta)).date())
    date_range = pd.date_range(start=start_date, end=end_date, freq="d")
    group = group.reindex(date_range)
    group.region = group.region.ffill()
    return group


def preprocess_data(df: pd.DataFrame, region: str | None = None) -> pd.DataFrame:
    df = df.copy()

    df["region"] = df.place.str.split(", ", expand=True)[1]
    df.region = df.region.fillna(df.place)
    df.region = df.region.replace({"CA": "California", "B.C.": "Baja California"})

    df.time = pd.to_datetime(df.time)
    df.time = df.time.dt.tz_localize(None)
    df = df.sort_values("time")
    df = df.set_index("time")

    df = df[["depth", "mag", "region"]]

    df = df.groupby("region").resample("d").mean().reset_index()
    df = df.set_index("time")

    if region is None:
        regions = get_regions()
        df = df.loc[df.region.isin(regions)]
        df = (
            df.groupby("region")[["region", "mag", "depth"]]
            .apply(lambda group: reindex(group, 0), include_groups=False)
            .reset_index(0, drop=True)
        )
        df.mag = df.groupby("region").mag.ffill()
        df.depth = df.groupby("region").depth.ffill()
    else:
        df = df.loc[df.region == region]
        start_date = df.index.min()
        end_date = pd.Timestamp(datetime.today().date())
        date_range = pd.date_range(start=start_date, end=end_date, freq="d")
        df = df.reindex(date_range)
        df.region = df.region.ffill()
        df.mag = df.mag.ffill()
        df.depth = df.depth.ffill()

    return df


def create_features(df: pd.DataFrame, region: str | None, horizon: int) -> pd.DataFrame:
    df = df.copy()

    if region is None:
        regions = get_regions()
        df = df.loc[df.region.isin(regions)]
        df = (
            df.groupby("region")[["region", "mag", "depth"]]
            .apply(lambda group: reindex(group, horizon), include_groups=False)
            .reset_index(0, drop=True)
        )
    else:
        start_date = df.index.min()
        end_date = pd.Timestamp((datetime.now() + timedelta(days=horizon)).date())
        date_range = pd.date_range(start=start_date, end=end_date, freq="d")
        df = df.reindex(date_range)
        df.region = df.region.ffill()

    df["day"] = df.index.day
    df["dayofweek"] = df.index.dayofweek
    df["dayofyear"] = df.index.dayofyear

    for i in range(1, _LAGS + 1):
        df[f"mag_lag_{i}"] = df.groupby("region").mag.shift(i)

    for i in range(1, _LAGS + 1):
        df[f"depth_lag_{i}"] = df.groupby("region").depth.shift(i)

    df["mag_ewma"] = df.groupby("region")["mag"].transform(lambda x: x.ewm(span=7, adjust=False).mean())
    df["depth_ewma"] = df.groupby("region")["depth"].transform(lambda x: x.ewm(span=7, adjust=False).mean())

    return df


def add_confidence_intervals(df) -> pd.DataFrame:
    df = df.copy()
    today = pd.Timestamp.now()
    df_past = df.loc[df.Date <= today].copy()
    df_future = df.loc[df.Date > today].copy()

    df_past.loc[:, "horizon"] = np.nan
    df_future.loc[:, "horizon"] = np.arange(1, len(df_future) + 1)

    magnitude_error = df_past["Magnitude"] - df_past["Magnitude Forecast"]
    depth_error = df_past["Depth"] - df_past["Depth Forecast"]
    magnitude_std = np.std(magnitude_error, ddof=1)
    depth_std = np.std(depth_error, ddof=1)

    z_90 = norm.ppf(0.95)
    z_50 = norm.ppf(0.75)

    df_past.loc[:, "magnitude_std"] = magnitude_std
    df_past.loc[:, "depth_std"] = depth_std

    df_past.loc[:, "Lower 90 Magnitude Forecast"] = df_past["Magnitude Forecast"] - z_90 * df_past["magnitude_std"]
    df_past.loc[:, "Upper 90 Magnitude Forecast"] = df_past["Magnitude Forecast"] + z_90 * df_past["magnitude_std"]

    df_past.loc[:, "Lower 90 Depth Forecast"] = df_past["Depth Forecast"] - z_90 * df_past["depth_std"]
    df_past.loc[:, "Upper 90 Depth Forecast"] = df_past["Depth Forecast"] + z_90 * df_past["depth_std"]

    df_past.loc[:, "Lower 50 Magnitude Forecast"] = df_past["Magnitude Forecast"] - z_50 * df_past["magnitude_std"]
    df_past.loc[:, "Upper 50 Magnitude Forecast"] = df_past["Magnitude Forecast"] + z_50 * df_past["magnitude_std"]

    df_past.loc[:, "Lower 50 Depth Forecast"] = df_past["Depth Forecast"] - z_50 * df_past["depth_std"]
    df_past.loc[:, "Upper 50 Depth Forecast"] = df_past["Depth Forecast"] + z_50 * df_past["depth_std"]

    df_future.loc[:, "magnitude_std"] = magnitude_std * np.sqrt(df_future["horizon"])
    df_future.loc[:, "depth_std"] = depth_std * np.sqrt(df_future["horizon"])

    df_future.loc[:, "Lower 90 Magnitude Forecast"] = (
        df_future["Magnitude Forecast"] - z_90 * df_future["magnitude_std"]
    )
    df_future.loc[:, "Upper 90 Magnitude Forecast"] = (
        df_future["Magnitude Forecast"] + z_90 * df_future["magnitude_std"]
    )

    df_future.loc[:, "Lower 90 Depth Forecast"] = df_future["Depth Forecast"] - z_90 * df_future["depth_std"]
    df_future.loc[:, "Upper 90 Depth Forecast"] = df_future["Depth Forecast"] + z_90 * df_future["depth_std"]

    df_future.loc[:, "Lower 50 Magnitude Forecast"] = (
        df_future["Magnitude Forecast"] - z_50 * df_future["magnitude_std"]
    )
    df_future.loc[:, "Upper 50 Magnitude Forecast"] = (
        df_future["Magnitude Forecast"] + z_50 * df_future["magnitude_std"]
    )

    df_future.loc[:, "Lower 50 Depth Forecast"] = df_future["Depth Forecast"] - z_50 * df_future["depth_std"]
    df_future.loc[:, "Upper 50 Depth Forecast"] = df_future["Depth Forecast"] + z_50 * df_future["depth_std"]

    df_future.loc[:, "Magnitude"] = np.nan
    df_future.loc[:, "Depth"] = np.nan

    return pd.concat([df_past, df_future], axis=0)


def add_confidence_intervals_per_region(df: pd.DataFrame, region: str | None = None) -> pd.DataFrame:
    if region is None:
        return (
            df.groupby("Region")[
                [
                    "Date",
                    "Magnitude",
                    "Depth",
                    "Region",
                    "Magnitude Forecast",
                    "Depth Forecast",
                    "Latitude",
                    "Longitude",
                ]
            ]
            .apply(add_confidence_intervals)
            .reset_index(drop=True)
        )
    return add_confidence_intervals(df)


def get_forecast(region: str | None = None) -> pd.DataFrame:
    model = load_model()
    df = get_recent_earthquakes()
    df = preprocess_data(df, region)

    # Forecast earthquakes for 3 days
    for horizon in range(1, 3 + 1):
        df = create_features(df, region, horizon)
        features = (
            [
                "day",
                "dayofweek",
                "dayofyear",
                "mag_ewma",
                "depth_ewma",
            ]
            + [f"mag_lag_{i}" for i in range(1, _LAGS + 1)]
            + [f"depth_lag_{i}" for i in range(1, _LAGS + 1)]
        )
        cat_features = ["region"]
        forecast = model.predict(df[features + cat_features])
        df["mag_forecast"] = forecast[:, 0]
        df["depth_forecast"] = forecast[:, 1]
        df.mag = df.mag.fillna(df.mag_forecast)
        df.depth = df.depth.fillna(df.depth_forecast)

    df = df.reset_index()
    df = df[["index", "mag", "mag_forecast", "depth", "depth_forecast", "region"]]
    df["Latitude"] = df.region.map(_MEDIAN_LATITUDE)
    df["Longitude"] = df.region.map(_MEDIAN_LONGITUDE)
    df = df.rename(
        columns={
            "index": "Date",
            "mag": "Magnitude",
            "depth": "Depth",
            "region": "Region",
            "mag_forecast": "Magnitude Forecast",
            "depth_forecast": "Depth Forecast",
        }
    )
    df = add_confidence_intervals_per_region(df, region)
    df = df[
        [
            "Date",
            "Region",
            "Latitude",
            "Longitude",
            "Magnitude",
            "Magnitude Forecast",
            "Lower 90 Magnitude Forecast",
            "Upper 90 Magnitude Forecast",
            "Lower 50 Magnitude Forecast",
            "Upper 50 Magnitude Forecast",
            "Depth",
            "Depth Forecast",
            "Lower 90 Depth Forecast",
            "Upper 90 Depth Forecast",
            "Lower 50 Depth Forecast",
            "Upper 50 Depth Forecast",
        ]
    ]
    date = pd.Timestamp.now() - pd.Timedelta(days=7)
    return df.loc[df.Date >= date]


def forecast_earthquakes() -> pd.DataFrame:
    df = get_forecast()
    today = pd.Timestamp.now()
    df = df.loc[df.Date > today]
    df = df[
        [
            "Date",
            "Region",
            "Latitude",
            "Longitude",
            "Magnitude Forecast",
            "Lower 90 Magnitude Forecast",
            "Upper 90 Magnitude Forecast",
            "Lower 50 Magnitude Forecast",
            "Upper 50 Magnitude Forecast",
            "Depth Forecast",
            "Lower 90 Depth Forecast",
            "Upper 90 Depth Forecast",
            "Lower 50 Depth Forecast",
            "Upper 50 Depth Forecast",
        ]
    ]
    return df
