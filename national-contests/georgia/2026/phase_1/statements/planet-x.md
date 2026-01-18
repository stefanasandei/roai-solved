# Planet X Model Selection

## Overview
On Planet X, scientists have been attempting to predict a mysterious phenomenon, **Y**, based on features **X**. However, Planet X experiences unusual gravitational anomalies that occasionally damage measuring instruments during data collection.

Over the years, they have collected data and trained 1000 regression models. Unfortunately, only a small number of these models (**exactly 5**) were trained while the instruments were functioning correctly. The rest were trained on corrupted labels.

## Task
Your goal is to identify the indices of the 5 correctly trained models.

## Input Files
### `train_data.csv`
Tabular data containing:
- The first 15 columns: Features used to train the models.
- The remaining 1000 columns: Predictions from 1000 models on the training data.

You can load the data as follows:

```python
import pandas as pd

data = pd.read_csv("train_data.csv")
features = data.iloc[:, :15]
predictions = data.iloc[:, 15:]
```

## Output Format
You must create a CSV file containing exactly **three columns and one row**:

```csv
subtaskID,datapointID,answer
1,0,"1,4,3,16,246"
```

Where:
- `subtaskID`: Exactly 1 (required by the platform format).
- `datapointID`: Exactly 0.
- `answer`: A string containing 5 comma-separated model indices (0-999).

**Do not change the column names under any circumstances.**

## Evaluation
Your submission will be evaluated based on Accuracy.

### Scoring Rules
The final score is based on a 100-point accuracy scale. Points are awarded as follows:
- 1 model identified: 20 points
- 2 models identified: 40 points
- 3 models identified: 60 points
- 4 models identified: 80 points
- 5 models identified: 100 points

Good luck, Earth scientist!