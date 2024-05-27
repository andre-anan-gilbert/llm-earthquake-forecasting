import os

import catboost as cb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Loading data.")
data = os.path.join(os.path.dirname(__file__), "earthquakes.csv")
df = pd.read_csv(data)
print("Loaded data.")

print("Preprocessing data.")
df.time = pd.to_datetime(df.time)
df = df.loc[df.time >= "1994-01-01"]
df = df.sort_values("time")
df = df.set_index("time")

df["region"] = df.place.str.split(", ", expand=True)[1]
df.region = df.region.fillna(df.place)
df.region = df.region.replace("CA", "California")

df.latitude = df.latitude.astype("float32")
df.longitude = df.longitude.astype("float32")
df.depth = df.depth.astype("float32")
df.mag = df.mag.astype("float32")

df = df[["latitude", "longitude", "depth", "mag", "region"]]

df = df.groupby("region").resample("h").mean()
df = df.reset_index()
df.mag = df.mag.fillna(-1.0)

df["hour"] = df.time.dt.hour
df["month"] = df.time.dt.month
df["dayofmonth"] = df.time.dt.day

df["mag_shifted_96h"] = df.groupby("region").mag.shift(96)
df["mag_shifted_120h"] = df.groupby("region").mag.shift(120)
df["mag_shifted_168h"] = df.groupby("region").mag.shift(168)

df["depth_shifted_96h"] = df.groupby("region").depth.shift(96)
df["depth_shifted_120h"] = df.groupby("region").depth.shift(120)
df["depth_shifted_168h"] = df.groupby("region").depth.shift(168)

df["mag_rolling_mean_96h"] = df.groupby("region").mag.transform(
    lambda x: x.rolling(96).mean()
)
df["mag_rolling_mean_120h"] = df.groupby("region").mag.transform(
    lambda x: x.rolling(120).mean()
)
df["mag_rolling_mean_168h"] = df.groupby("region").mag.transform(
    lambda x: x.rolling(168).mean()
)
df["mag_rolling_std_96h"] = df.groupby("region").mag.transform(
    lambda x: x.rolling(96).std()
)
df["mag_rolling_std_120h"] = df.groupby("region").mag.transform(
    lambda x: x.rolling(120).std()
)
df["mag_rolling_std_168h"] = df.groupby("region").mag.transform(
    lambda x: x.rolling(168).std()
)

df["depth_rolling_mean_96h"] = df.groupby("region").depth.transform(
    lambda x: x.rolling(96).mean()
)
df["depth_rolling_mean_120h"] = df.groupby("region").depth.transform(
    lambda x: x.rolling(120).mean()
)
df["depth_rolling_mean_168h"] = df.groupby("region").depth.transform(
    lambda x: x.rolling(168).mean()
)
df["depth_rolling_std_96h"] = df.groupby("region").depth.transform(
    lambda x: x.rolling(96).std()
)
df["depth_rolling_std_120h"] = df.groupby("region").depth.transform(
    lambda x: x.rolling(120).std()
)
df["depth_rolling_std_168h"] = df.groupby("region").depth.transform(
    lambda x: x.rolling(168).std()
)
print("Preprocessed data.")

features = [
    "dayofmonth",
    "hour",
    "month",
    "mag_shifted_96h",
    "mag_shifted_120h",
    "mag_shifted_168h",
    "depth_shifted_96h",
    "depth_shifted_120h",
    "depth_shifted_168h",
    "mag_rolling_mean_96h",
    "mag_rolling_mean_120h",
    "mag_rolling_mean_168h",
    "mag_rolling_std_96h",
    "mag_rolling_std_120h",
    "mag_rolling_std_168h",
    "depth_rolling_mean_96h",
    "depth_rolling_mean_120h",
    "depth_rolling_mean_168h",
    "depth_rolling_std_96h",
    "depth_rolling_std_120h",
    "depth_rolling_std_168h",
]
cat_features = ["region"]
target = "mag"


def split_train_test(df, test_size):
    df_train = []
    df_test = []
    grouped = df.groupby("region")
    for _, group in grouped:
        n = len(group)
        test_index = int(n * (1 - test_size))
        df_train.append(group.iloc[:test_index])
        df_test.append(group.iloc[test_index:])
    df_train = pd.concat(df_train)
    df_test = pd.concat(df_test)
    return df_train, df_test


df_train, df_test = split_train_test(df, test_size=0.2)


print("Training model.")
model = cb.CatBoostRegressor(
    early_stopping_rounds=50,
    cat_features=cat_features,
    depth=7,
    save_snapshot=True,
)
model.fit(df_train[features + cat_features], df_train[target])

prediction = model.predict(df_test[features + cat_features])
print(f"Mean Absolute Error: {mean_absolute_error(df_test[target], prediction)}")
print(
    f"Root Mean Squared Error: {np.sqrt(mean_squared_error(df_test[target], prediction))}"
)
print(f"R2 Score: {r2_score(df_test[target], prediction)}")

live_data = pd.read_csv(
    "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&eventtype=earthquake&limit=20000"
)
live_data.time = pd.to_datetime(live_data.time)
live_data = live_data.sort_values("time")
live_data = live_data.set_index("time")

live_data["region"] = live_data.place.str.split(", ", expand=True)[1]
live_data.region = live_data.region.fillna(live_data.place)
live_data.region = live_data.region.replace("CA", "California")

live_data.latitude = live_data.latitude.astype("float32")
live_data.longitude = live_data.longitude.astype("float32")
live_data.depth = live_data.depth.astype("float32")
live_data.mag = live_data.mag.astype("float32")

live_data = live_data[["latitude", "longitude", "depth", "mag", "region"]]

live_data = live_data.groupby("region").resample("h").mean()
live_data = live_data.reset_index()
live_data.mag = live_data.mag.fillna(-1.0)

live_data["hour"] = live_data.time.dt.hour
live_data["month"] = live_data.time.dt.month
live_data["dayofmonth"] = live_data.time.dt.day

live_data["mag_shifted_96h"] = live_data.groupby("region").mag.shift(96)
live_data["mag_shifted_120h"] = live_data.groupby("region").mag.shift(120)
live_data["mag_shifted_168h"] = live_data.groupby("region").mag.shift(168)

live_data["depth_shifted_96h"] = live_data.groupby("region").depth.shift(96)
live_data["depth_shifted_120h"] = live_data.groupby("region").depth.shift(120)
live_data["depth_shifted_168h"] = live_data.groupby("region").depth.shift(168)

live_data["mag_rolling_mean_96h"] = live_data.groupby("region").mag.transform(
    lambda x: x.rolling(96).mean()
)
live_data["mag_rolling_mean_120h"] = live_data.groupby("region").mag.transform(
    lambda x: x.rolling(120).mean()
)
live_data["mag_rolling_mean_168h"] = live_data.groupby("region").mag.transform(
    lambda x: x.rolling(168).mean()
)
live_data["mag_rolling_std_96h"] = live_data.groupby("region").mag.transform(
    lambda x: x.rolling(96).std()
)
live_data["mag_rolling_std_120h"] = live_data.groupby("region").mag.transform(
    lambda x: x.rolling(120).std()
)
live_data["mag_rolling_std_168h"] = live_data.groupby("region").mag.transform(
    lambda x: x.rolling(168).std()
)

live_data["depth_rolling_mean_96h"] = live_data.groupby("region").depth.transform(
    lambda x: x.rolling(96).mean()
)
live_data["depth_rolling_mean_120h"] = live_data.groupby("region").depth.transform(
    lambda x: x.rolling(120).mean()
)
live_data["depth_rolling_mean_168h"] = live_data.groupby("region").depth.transform(
    lambda x: x.rolling(168).mean()
)
live_data["depth_rolling_std_96h"] = live_data.groupby("region").depth.transform(
    lambda x: x.rolling(96).std()
)
live_data["depth_rolling_std_120h"] = live_data.groupby("region").depth.transform(
    lambda x: x.rolling(120).std()
)
live_data["depth_rolling_std_168h"] = live_data.groupby("region").depth.transform(
    lambda x: x.rolling(168).std()
)

live_prediction = model.predict(live_data[features + cat_features])
print(f"Mean Absolute Error: {mean_absolute_error(live_data[target], live_prediction)}")
print(
    f"Root Mean Squared Error: {np.sqrt(mean_squared_error(live_data[target], live_prediction))}"
)
print(f"R2 Score: {r2_score(live_data[target], live_prediction)}")

model_file = os.path.join(os.path.dirname(__file__), "model")
model.save_model(model_file)
