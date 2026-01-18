# Truck APS Fault Data Optimization Challenge

## Problem Description

Astamakha is a hardworking individual tasked with analyzing Air Pressure System (APS) faults in heavy trucks. His superiors require his system to achieve an F1 score of at least 0.92. His friend Giorgi, who enjoys complex challenges, suggested: "Let's achieve F1 >= 0.92 using as little data as possible!"

Astamakha accepted the challenge but decided to take a snack break first. Unfortunately, while he was away, his cat walked across the keyboard, introducing numerous errors into the data!

**Your task is to help Astamakha complete the assignment and the challenge using the minimum amount of data.**

## Task

You are provided with training data (`train_data.csv`). You must select a subset of **rows** and **columns/features** that:

1.  Achieves an **F1 score >= 0.92** on both public and private test sets. (Higher scores do not provide additional benefits; the goal is simply to cross this threshold).
2.  Uses the **minimum amount of data** (fewer rows and columns result in a better score).

# Provided Data

`train_data.csv` - Data with a shape of `(2870, 195)`, where 194 are features and the last column is the `class` label.

```python
import pandas as pd

train_data = pd.read_csv("train_data.csv")
```

## Submission Format

Create a JSON file with the following structure:

```json
{
  "rows": [0, 1, 5, 10, ...],
  "columns": [91, 124, 181, ...]
}
```

-   **rows**: A list of row indices (starting from 0) from `train_data.csv` to be used.
-   **columns**: A list of column indices (starting from 0) or column names (excluding the 'class' column).

## Evaluation System

Based on your submitted JSON file, we will create a subset of the original training data and train a model using the exact code below:

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced",
        max_depth=None,
        min_samples_leaf=1,
    )),
])
```

**Note**: The model `seed` is fixed at 42 and cannot be changed.

## Scoring

### F1 Threshold (Pass/Fail)

Your submission must satisfy the following condition:

`F1_Test >= 0.92`

If the F1 threshold is **not met**, your score will be **0**.

### Compression Score

If the F1 threshold is crossed, your score is calculated as follows:

$$ score = 0.5 \cdot \frac{\log(r / r_{now})}{\log(r)} + 0.5 \cdot \frac{\log(c / c_{now})}{\log(c)} $$

Where:
-   $r$ = Total number of rows in the original training data (2870)
-   $r_{now}$ = Number of rows in your selection
-   $c$ = Total number of feature columns (194)
-   $c_{now}$ = Number of columns in your selection

**The higher the score, the better.** The score rewards the use of fewer rows and columns. This score is scaled from 0 to 100, where 100 represents the best possible solution found by the organizers.

### Scoring Examples

| Rows | Cols | F1 Threshold | Score |
| :--- | :--- | :----------- | :---- |
| 2870 | 194  | PASS         | 0.000 |
| 2870 | 6    | PASS         | 0.330 |
| 300  | 6    | FAIL         | 0.000 |

## Good Luck!

Help Astamakha impress Giorgi with the most compressed dataset that still achieves F1 >= 0.92!