# Memory Trace

## Overview
You are provided with:
1.  **Tabular data** (`train_data.csv`)
2.  **Two trained machine learning models**:
    *   `model_A.joblib`
    *   `model_B.joblib`

Each model was trained on **exactly half** of the data:
-   These halves are non-overlapping.
-   Each row (record) participated in the training set of only one model.
-   Both models solve a binary classification task.

**Your goal:** For each row in the data, determine which model used it during its training process.

## Task
Make a prediction for each row in the dataset:
-   **0** → The row was used to train **Model A**.
-   **1** → The row was used to train **Model B**.

## Input Files
### `train_data.csv`
Tabular data containing:
-   `row_id` – A unique identifier for each row.
-   Feature columns – Both numerical and categorical.

### `model_A.joblib`, `model_B.joblib`
Two pre-trained `sklearn` models saved in `joblib` format. You can load them as follows:

```python
import joblib

model_A = joblib.load("model_A.joblib")
model_B = joblib.load("model_B.joblib")
```

## Output Format
You must create a CSV file containing exactly three columns:

```csv
subtaskID,datapointID,answer
1,0,0
1,1,1
1,2,0
...
```

Where:
-   `subtaskID`: Constantly 1 (required by platform format).
-   `datapointID`: Matches the IDs from the input database.
-   `answer`: $\in {0, 1}$
    -   **0** = Model A
    -   **1** = Model B

**Do not change the column names under any circumstances.** Every row must be represented exactly once in the file.

## Evaluation
Your work will be evaluated on a scale of 0–100.

### Scoring Rules
The final score is calculated using the following formula:

$$ Score = 100 \times \frac{Accuracy - 0.5}{BestSolution - 0.5} $$

Or in code form:

```python
score = 100.0 * (acc - 0.5) / (best_solution - 0.5)
```

-   Random guessing (≈ 50% accuracy) → **0 points**
-   Best solution → **100 points**
-   Intermediate results will be evaluated using linear scaling.

Only the final score is decisive.

## Constraints & Rules
1.  **Prohibited**: Retraining the provided models.
2.  **Prohibited**: Modifying the data.