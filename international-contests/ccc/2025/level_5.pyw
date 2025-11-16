import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import re
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = "Level 5"
LEVEL_5_FILE = os.path.join(DATA_DIR, "level_5.in")

try:
    full_df = pd.read_csv(LEVEL_5_FILE, low_memory=False)
    clean_columns = {
        col: re.sub(r"[^A-Za-z0-9_]+", "_", col) for col in full_df.columns
    }
    full_df.rename(columns=clean_columns, inplace=True)
    full_df["Arrivals"] = pd.to_numeric(full_df["Arrivals"], errors="coerce")
except FileNotFoundError:
    print(f"Error: Could not find '{LEVEL_5_FILE}'")
    exit()

full_df = full_df.sort_values(by=["BOP", "Day"]).reset_index(drop=True)

full_df["day_of_year"] = ((full_df["Day"] - 1) % 365) + 1
full_df["day_sin"] = np.sin(2 * np.pi * full_df["day_of_year"] / 365)
full_df["day_cos"] = np.cos(2 * np.pi * full_df["day_of_year"] / 365)

for lag in [1, 2, 3, 7, 14, 30]:
    full_df[f"lag_{lag}"] = full_df.groupby("BOP")["Arrivals"].shift(lag)

for window in [3, 7, 14, 30]:
    full_df[f"rmean_{window}"] = full_df.groupby("BOP")["Arrivals"].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    full_df[f"rstd_{window}"] = full_df.groupby("BOP")["Arrivals"].transform(
        lambda x: x.rolling(window, min_periods=1).std()
    )

full_df["vel"] = full_df.groupby("BOP")["Arrivals"].diff()
full_df["accel"] = full_df.groupby("BOP")["vel"].diff()

full_df["ema_7"] = full_df.groupby("BOP")["Arrivals"].transform(
    lambda x: x.ewm(span=7, adjust=False).mean()
)
full_df["ema_14"] = full_df.groupby("BOP")["Arrivals"].transform(
    lambda x: x.ewm(span=14, adjust=False).mean()
)

full_df["wind_mag"] = np.sqrt(full_df["Wind_X_m_s_"] ** 2 + full_df["Wind_Y_m_s_"] ** 2)
full_df["wind_dir"] = np.arctan2(full_df["Wind_Y_m_s_"], full_df["Wind_X_m_s_"])

full_df["occ_arrivals"] = full_df["Occupancy"] / (full_df["Arrivals"] + 1)

# ----
train_df = full_df[full_df["Day"] <= 730].copy()
test_df = full_df[full_df["Day"] > 730].copy()

train_arrivals = train_df["Arrivals"].fillna(0)
train_mean = train_arrivals.mean()
train_std = train_arrivals.std() + 1e-5

bop_mean_dict = (
    train_df.groupby("BOP")["Arrivals"].apply(lambda x: x.fillna(0).mean()).to_dict()
)

day_mean_dict = (
    train_df.groupby("day_of_year")["Arrivals"]
    .apply(lambda x: x.fillna(0).mean())
    .to_dict()
)

bop_day_stats = (
    train_df.groupby(["BOP", "day_of_year"])["Arrivals"]
    .agg(["mean", "std"])
    .fillna(0)
    .reset_index()
)
bop_day_mean = bop_day_stats.set_index(["BOP", "day_of_year"])["mean"].to_dict()
bop_day_std = bop_day_stats.set_index(["BOP", "day_of_year"])["std"].to_dict()

full_df["occ_quartile"] = pd.qcut(
    full_df["Occupancy"], q=4, labels=False, duplicates="drop"
)
train_occ_q = pd.qcut(train_df["Occupancy"], q=4, labels=False, duplicates="drop")
occ_q_mean = (
    train_df.groupby(train_occ_q)["Arrivals"]
    .apply(lambda x: x.fillna(0).mean())
    .to_dict()
)


full_df["bop_mean"] = full_df["BOP"].map(bop_mean_dict).fillna(train_mean)
full_df["day_mean"] = full_df["day_of_year"].map(day_mean_dict).fillna(train_mean)

full_df["bop_day_mean"] = full_df.apply(
    lambda r: bop_day_mean.get(
        (r["BOP"], r["day_of_year"]), bop_mean_dict.get(r["BOP"], train_mean)
    ),
    axis=1,
)
full_df["bop_day_std"] = full_df.apply(
    lambda r: bop_day_std.get((r["BOP"], r["day_of_year"]), train_std), axis=1
)

full_df["occ_quartile_mean"] = (
    full_df["occ_quartile"].map(occ_q_mean).fillna(train_mean)
)

full_df = full_df.fillna(0)

train_data = full_df[full_df["Day"] <= 730]
test_data = full_df[full_df["Day"] > 730].copy()


features = [
    "BOP",
    "Occupancy",
    "Wind_X_m_s_",
    "Wind_Y_m_s_",
    "Insects_Delta_g_m_",
    "day_of_year",
    "day_sin",
    "day_cos",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "lag_30",
    "rmean_3",
    "rmean_7",
    "rmean_14",
    "rmean_30",
    "rstd_3",
    "rstd_7",
    "rstd_14",
    "rstd_30",
    "vel",
    "accel",
    "ema_7",
    "ema_14",
    "wind_mag",
    "wind_dir",
    "bop_mean",
    "day_mean",
    "bop_day_mean",
    "bop_day_std",
    "occ_arrivals",
    "occ_quartile_mean",
]

X_train = train_data[features]
y_train = train_data["Arrivals"]
X_test = test_data[features]

params = {
    "objective": "regression_l1",
    "metric": "mae",
    "n_estimators": 3000,
    "learning_rate": 0.012,
    "feature_fraction": 0.65,
    "bagging_fraction": 0.65,
    "bagging_freq": 1,
    "lambda_l1": 2.0,
    "lambda_l2": 2.0,
    "num_leaves": 80,
    "max_depth": 12,
    "min_child_samples": 3,
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1,
}

model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train)

test_data["pred"] = model.predict(X_test)

results = []
for day in range(731, 761):
    day_data = test_data[test_data["Day"] == day]
    top_50 = day_data.nlargest(50, "pred")["BOP"].astype(str).tolist()
    results.append({"Day": day, "Top 50 Arrivals BOPs": " ".join(top_50)})

pd.DataFrame(results).to_csv("level_5_submission.csv", index=False)
print(f"Submission saved with {len(features)} features")
