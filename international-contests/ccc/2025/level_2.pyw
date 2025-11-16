import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

level1_df = pd.read_csv("level_2/all_data_from_level_1.in")
level1_df["Temperature [°C]"] = pd.to_numeric(
    level1_df["Temperature [°C]"], errors="coerce"
)
level1_df["Humidity [%]"] = pd.to_numeric(level1_df["Humidity [%]"], errors="coerce")
level1_df["BOP"] = pd.to_numeric(level1_df["BOP"], errors="coerce")

# correct temperatures: if > 50, assume F & convert to C
level1_df["Temperature [°C]"] = level1_df["Temperature [°C]"].apply(
    lambda x: (x - 32) * 5 / 9 if x > 50 else x
)


# collect all training data from all level 2 files
all_train_dfs = []
for level in ["level_2_a.in", "level_2_b.in", "level_2_c.in"]:
    level2_df = pd.read_csv(f"level_2/{level}")
    level2_df["BOP"] = pd.to_numeric(level2_df["BOP"], errors="coerce")
    level2_df["Vegetation [%]"] = pd.to_numeric(
        level2_df["Vegetation [%]"], errors="coerce"
    )
    level2_df["Insects [g/m²]"] = pd.to_numeric(
        level2_df["Insects [g/m²]"], errors="coerce"
    )
    level2_df["Urban Light [%]"] = pd.to_numeric(
        level2_df["Urban Light [%]"], errors="coerce"
    )
    level2_df["Bird Love Score [<3]"] = (
        level2_df["Bird Love Score [<3]"].replace("missing", np.nan).astype(float)
    )

    # merge with level 1
    merged_df = pd.merge(level2_df, level1_df, on="BOP", how="left")
    all_train_dfs.append(merged_df)

# combine all training data
combined_df = pd.concat(all_train_dfs, ignore_index=True)
train_df = combined_df.dropna(subset=["Bird Love Score [<3]"])
X_train = train_df[
    [
        "Temperature [°C]",
        "Humidity [%]",
        "Vegetation [%]",
        "Insects [g/m²]",
        "Urban Light [%]",
    ]
]
y_train = train_df["Bird Love Score [<3]"]

# train model on all available data
model = LinearRegression()
model.fit(X_train, y_train)

# calculate overall RMSE
y_pred_train = model.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f"Overall Training RMSE: {rmse}")


# function to process each level 2 file for predictions
def process_level2(level2_file):
    level2_df = pd.read_csv(f"level_2/{level2_file}")
    level2_df["BOP"] = pd.to_numeric(level2_df["BOP"], errors="coerce")
    level2_df["Vegetation [%]"] = pd.to_numeric(
        level2_df["Vegetation [%]"], errors="coerce"
    )
    level2_df["Insects [g/m²]"] = pd.to_numeric(
        level2_df["Insects [g/m²]"], errors="coerce"
    )
    level2_df["Urban Light [%]"] = pd.to_numeric(
        level2_df["Urban Light [%]"], errors="coerce"
    )
    level2_df["Bird Love Score [<3]"] = (
        level2_df["Bird Love Score [<3]"].replace("missing", np.nan).astype(float)
    )

    merged_df = pd.merge(level2_df, level1_df, on="BOP", how="left")

    # predict missing
    missing_df = merged_df[merged_df["Bird Love Score [<3]"].isna()]
    if not missing_df.empty:
        X_missing = missing_df[
            [
                "Temperature [°C]",
                "Humidity [%]",
                "Vegetation [%]",
                "Insects [g/m²]",
                "Urban Light [%]",
            ]
        ]
        predictions = model.predict(X_missing)

        # output CSV
        output_df = pd.DataFrame(
            {"BOP": missing_df["BOP"], "Bird Love Score [<3]": predictions}
        )
        output_file = f"{level2_file.replace('.in', '_predictions.csv')}"
        output_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")


for level in ["level_2_a.in", "level_2_b.in", "level_2_c.in"]:
    process_level2(level)
