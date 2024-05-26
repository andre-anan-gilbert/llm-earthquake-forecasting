import os

import catboost as cb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = os.path.join(os.path.dirname(__file__), "earthquakes.csv")
df = pd.read_csv(data)

df.time = pd.to_datetime(df.time)
df = df.loc[df.time >= "2004-01-01"]
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

df["mag_shifted_12h"] = df.groupby("region").mag.shift(12)
df["mag_shifted_24h"] = df.groupby("region").mag.shift(24)

df["depth_shifted_12h"] = df.groupby("region").depth.shift(12)
df["depth_shifted_24h"] = df.groupby("region").depth.shift(24)

df["mag_rolling_mean_12h"] = df.groupby("region").mag.transform(
    lambda x: x.rolling(12).mean()
)
df["mag_rolling_mean_24h"] = df.groupby("region").mag.transform(
    lambda x: x.rolling(24).mean()
)
df["mag_rolling_std_12h"] = df.groupby("region").mag.transform(
    lambda x: x.rolling(12).std()
)
df["mag_rolling_std_24h"] = df.groupby("region").mag.transform(
    lambda x: x.rolling(24).std()
)

df["depth_rolling_mean_12h"] = df.groupby("region").depth.transform(
    lambda x: x.rolling(12).mean()
)
df["depth_rolling_mean_24h"] = df.groupby("region").depth.transform(
    lambda x: x.rolling(24).mean()
)
df["depth_rolling_std_12h"] = df.groupby("region").depth.transform(
    lambda x: x.rolling(12).std()
)
df["depth_rolling_std_24h"] = df.groupby("region").depth.transform(
    lambda x: x.rolling(24).std()
)

features = [
    "dayofmonth",
    "hour",
    "month",
    "mag_shifted_12h",
    "mag_shifted_24h",
    "depth_shifted_12h",
    "depth_shifted_24h",
    "mag_rolling_mean_12h",
    "mag_rolling_mean_24h",
    "mag_rolling_std_12h",
    "mag_rolling_std_24h",
    "depth_rolling_mean_12h",
    "depth_rolling_mean_24h",
    "depth_rolling_std_12h",
    "depth_rolling_std_24h",
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

model = cb.CatBoostRegressor(
    early_stopping_rounds=20,
    cat_features=cat_features,
    iterations=500,
    learning_rate=0.1,
    depth=7,
)
model.fit(df_train[features + cat_features], df_train[target], verbose=True)

prediction = model.predict(df_test[features + cat_features])
print(f"Mean Absolute Error: {mean_absolute_error(df_test[target], prediction)}")
print(
    f"Root Mean Squared Error: {np.sqrt(mean_squared_error(df_test[target], prediction))}"
)

model_file = os.path.join(os.path.dirname(__file__), "model")
model.save_model(model_file)
