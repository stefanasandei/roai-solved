import pandas as pd

df = pd.read_csv("Level 1/level_1_a.in")
df["Temperature [°C]"] = pd.to_numeric(df["Temperature [°C]"], errors="coerce")
df["Humidity [%]"] = pd.to_numeric(df["Humidity [%]"], errors="coerce")
df["BOP"] = pd.to_numeric(df["BOP"], errors="coerce")

with open("a.txt", "w") as f:
    x = df.sort_values(
        by=["Temperature [°C]", "Humidity [%]", "BOP"], ascending=[False, True, True]
    )["BOP"].to_list()
    f.write(" ".join(map(str, x)))
