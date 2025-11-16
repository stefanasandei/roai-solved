import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = "Level 4"
LEVEL_1_FILE = os.path.join(DATA_DIR, "all_data_from_level_1.in")
LEVEL_4_FILE = os.path.join(DATA_DIR, "level_4.in")

try:
    level1_df = pd.read_csv(LEVEL_1_FILE)
    level1_df["Temperature [°C]"] = pd.to_numeric(
        level1_df["Temperature [°C]"], errors="coerce"
    )
    level1_df["BOP"] = pd.to_numeric(level1_df["BOP"], errors="coerce")
    level1_df["Temperature [°C]"] = level1_df["Temperature [°C]"].apply(
        lambda x: (x - 32) * 5 / 9 if x > 50 else x
    )
    bop_temps = level1_df.set_index("BOP")["Temperature [°C]"]
except FileNotFoundError:
    print(f"Error: Temperature data file '{LEVEL_1_FILE}' not found.")
    exit()


def analyze_path_metrics(path_list, temp_lookup):
    if not path_list:
        return None

    path = np.array(path_list)
    jumps = np.sum(np.abs(np.diff(path)) > 1) if len(path) > 1 else 0

    return {
        "symmetric": list(path) == list(path[::-1]),
        "circular": len(path) > 1 and path[0] == path[-1],
        "temps": [temp_lookup.get(bop, np.nan) for bop in path_list],
        "path_tuple": tuple(path_list),
        "path_length": len(path_list),
        "unique_bops": set(path),
        "jumps": jumps,
    }


try:
    level4_df = pd.read_csv(LEVEL_4_FILE)
    flock_profiles_data = {}
    flock_ids = sorted(level4_df["Flock ID"].unique())

    print("Generating features...")
    for flock_id in flock_ids:
        flock_data = level4_df[level4_df["Flock ID"] == flock_id]
        metrics_list = [
            analyze_path_metrics(list(map(int, str(p).split())), bop_temps)
            for p in flock_data["BOP Path"]
        ]
        metrics_list = [m for m in metrics_list if m]

        if metrics_list:
            num_paths = len(metrics_list)
            unique_paths = len(set(m["path_tuple"] for m in metrics_list))
            all_bops = set.union(*[m["unique_bops"] for m in metrics_list])
            all_temps = [t for m in metrics_list for t in m["temps"]]
            all_path_lengths = [m["path_length"] for m in metrics_list]

            flock_profiles_data[flock_id] = {
                "percent_symmetric": sum(m["symmetric"] for m in metrics_list)
                / num_paths,
                "percent_circular": sum(m["circular"] for m in metrics_list)
                / num_paths,
                "path_identical_ratio": 1.0
                - (unique_paths - 1) / max(num_paths - 1, 1),
                "avg_path_length": np.mean(all_path_lengths),
                "avg_temp": np.nanmean(all_temps),
                "std_path_length": np.std(all_path_lengths),
                "num_unique_bops_flock": len(all_bops),
                "flock_bop_range": max(all_bops) - min(all_bops) if all_bops else 0,
                "std_temp": np.nanstd(all_temps),
                "avg_jumps": np.mean([m["jumps"] for m in metrics_list]),
            }

    profiles_df = pd.DataFrame.from_dict(flock_profiles_data, orient="index")
    profiles_df.index.name = "Flock ID"
    profiles_df.fillna(0, inplace=True)

    known_species_df = (
        level4_df[level4_df["Species"] != "missing"]
        .drop_duplicates("Flock ID")
        .set_index("Flock ID")
    )
    profiles_df["Species"] = profiles_df.index.map(known_species_df["Species"])

    train_df = profiles_df.dropna(subset=["Species"])
    predict_df = profiles_df[profiles_df["Species"].isna()]

    X_train = train_df.drop("Species", axis=1)
    y_train = train_df["Species"]
    X_predict = predict_df.drop("Species", axis=1)

    print(f"Training data: {len(X_train)} flocks.")
    print(f"Data to predict: {len(X_predict)} flocks.")
    print(f"Features used: {list(X_train.columns)}")

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    predictions = model.predict(X_predict)

    submission_df = pd.DataFrame({"Flock ID": X_predict.index, "Species": predictions})
    submission_df.sort_values("Flock ID", inplace=True)

    output_file = "level_4_submission.csv"
    submission_df.to_csv(output_file, index=False)

except Exception as e:
    print(f"An error occurred: {e}")
