import os

import catboost as cb
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
df.region = df.region.replace("B.C.", "Baja California")

df = df[["depth", "mag", "region", "latitude", "longitude"]]

regions = df.region.value_counts()
top_k = 25
top_k_regions = regions.head(top_k).index
df = df.loc[df.region.isin(top_k_regions)]

df = df.groupby("region").resample("d").mean()
df = df.reset_index()
df.mag = df.mag.ffill()
df.depth = df.depth.ffill()
df.latitude = df.latitude.ffill()
df.longitude = df.longitude.ffill()

df["day"] = df.time.dt.day
df["dayofweek"] = df.time.dt.dayofweek
df["dayofyear"] = df.time.dt.dayofyear

start_lag = 3
end_lag = 10
for i in range(start_lag, end_lag + 1):
    df[f"mag_lag_{i}"] = df.groupby("region").mag.shift(i)

for i in range(start_lag, end_lag + 1):
    df[f"depth_lag_{i}"] = df.groupby("region").depth.shift(i)

for i in range(start_lag, end_lag + 1):
    df[f"latitude_lag_{i}"] = df.groupby("region").latitude.shift(i)

for i in range(start_lag, end_lag + 1):
    df[f"longitude_lag_{i}"] = df.groupby("region").longitude.shift(i)

df[f"mag_rolling_mean_{start_lag}"] = df.groupby("region").mag.transform(lambda x: x.rolling(window=start_lag).mean())
df[f"mag_rolling_std_{start_lag}"] = df.groupby("region").mag.transform(lambda x: x.rolling(window=start_lag).std())
df[f"mag_rolling_mean_{end_lag}"] = df.groupby("region").mag.transform(lambda x: x.rolling(window=end_lag).mean())
df[f"mag_rolling_std_{end_lag}"] = df.groupby("region").mag.transform(lambda x: x.rolling(window=end_lag).std())

df[f"depth_rolling_mean_{start_lag}"] = df.groupby("region").depth.transform(
    lambda x: x.rolling(window=start_lag).mean()
)
df[f"depth_rolling_std_{start_lag}"] = df.groupby("region").depth.transform(lambda x: x.rolling(window=start_lag).std())
df[f"depth_rolling_mean_{end_lag}"] = df.groupby("region").depth.transform(lambda x: x.rolling(window=end_lag).mean())
df[f"depth_rolling_std_{end_lag}"] = df.groupby("region").depth.transform(lambda x: x.rolling(window=end_lag).std())

df[f"latitude_rolling_mean_{start_lag}"] = df.groupby("region").latitude.transform(
    lambda x: x.rolling(window=start_lag).mean()
)
df[f"latitude_rolling_std_{start_lag}"] = df.groupby("region").latitude.transform(
    lambda x: x.rolling(window=start_lag).std()
)
df[f"latitude_rolling_mean_{end_lag}"] = df.groupby("region").latitude.transform(
    lambda x: x.rolling(window=end_lag).mean()
)
df[f"latitude_rolling_std_{end_lag}"] = df.groupby("region").latitude.transform(
    lambda x: x.rolling(window=end_lag).std()
)

df[f"longitude_rolling_mean_{start_lag}"] = df.groupby("region").longitude.transform(
    lambda x: x.rolling(window=start_lag).mean()
)
df[f"longitude_rolling_std_{start_lag}"] = df.groupby("region").longitude.transform(
    lambda x: x.rolling(window=start_lag).std()
)
df[f"longitude_rolling_mean_{end_lag}"] = df.groupby("region").longitude.transform(
    lambda x: x.rolling(window=end_lag).mean()
)
df[f"longitude_rolling_std_{end_lag}"] = df.groupby("region").longitude.transform(
    lambda x: x.rolling(window=end_lag).std()
)

print("Preprocessed data.")

features = (
    [
        "day",
        "dayofweek",
        "dayofyear",
        f"mag_rolling_mean_{start_lag}",
        f"mag_rolling_std_{start_lag}",
        f"depth_rolling_mean_{start_lag}",
        f"depth_rolling_std_{start_lag}",
        f"latitude_rolling_mean_{start_lag}",
        f"latitude_rolling_std_{start_lag}",
        f"longitude_rolling_mean_{start_lag}",
        f"longitude_rolling_std_{start_lag}",
        f"mag_rolling_mean_{end_lag}",
        f"mag_rolling_std_{end_lag}",
        f"depth_rolling_mean_{end_lag}",
        f"depth_rolling_std_{end_lag}",
        f"latitude_rolling_mean_{end_lag}",
        f"latitude_rolling_std_{end_lag}",
        f"longitude_rolling_mean_{end_lag}",
        f"longitude_rolling_std_{end_lag}",
    ]
    + [f"mag_lag_{i}" for i in range(start_lag, end_lag + 1)]
    + [f"depth_lag_{i}" for i in range(start_lag, end_lag + 1)]
    + [f"latitude_lag_{i}" for i in range(start_lag, end_lag + 1)]
    + [f"longitude_lag_{i}" for i in range(start_lag, end_lag + 1)]
)
cat_features = ["region"]
target = ["mag", "depth", "latitude", "longitude"]

n = len(df)
test_size = 0.2
index = int(n - (1 - test_size))
time = df.time.iloc[index]

df_train = df.loc[df.time < time]
df_test = df.loc[df.time >= time]

print("Training model.")
depth = 10
iterations = 1000
model = cb.CatBoostRegressor(
    early_stopping_rounds=20,
    cat_features=cat_features,
    depth=depth,
    iterations=iterations,
    loss_function="MultiRMSE",
)
model.fit(df_train[features + cat_features], df_train[target])

prediction = model.predict(df_test[features + cat_features])
print(f"Mean Absolute Error: {mean_absolute_error(df_test[target], prediction)}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(df_test[target], prediction))}")
print(f"R2 Score: {r2_score(df_test[target], prediction)}")

model_file = os.path.join(os.path.dirname(__file__), "multi_output_4_model_2")
model.save_model(model_file)
