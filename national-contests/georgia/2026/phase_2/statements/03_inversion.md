# "Inversion": Digital Data Restoration

https://judge.nitro-ai.org/competitions/gaia/ai-league-ii/3/view

## Context

Imagine you are working in a high-tech research center where a unique Artificial Intelligence has been created. This model accurately determines the final result of an experiment (\( y \)) based on ten different parameters.

Due to a technical failure on the server, one of the database columns was corrupted—all its values were replaced by zero. The model is "frozen," meaning retraining it is impossible. However, since the model already "knows" the relationship between parameters and the result, it becomes the key to restoring the information.

Your task is to restore the lost column values for each of the 1000 samples through mathematical optimization.

## Resources Provided

### 1. Model
File `model.pt` — a pre-trained TorchScript model. It takes 10 values and returns one "target" variable (\( y \)). To load the model, use:

```python
import torch
model = torch.jit.load("model.pt")
model.eval()
```

### 2. Data
File `data.csv` — contains 1000 records. The table provides 10 columns (from \( x\_0 \) to \( x\_9 \)) and the result \( y \). One of the columns is entirely filled with `0.0`s. This is your target.

## Task Essence

You must find values for the corrupted column that best align with the model's logic. In other words, if the restored values and the remaining 9 parameters are provided to the model, it should return a value as close as possible to the \( y \) provided in the table.

## Loading Data and Model

### Step 1 — Loading Data
```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

df = pd.read_csv("data.csv")
y = df["y"].values.astype("float32")
X = df.drop(columns="y").values.astype("float32")
```

### Step 2 — Loading and Verifying the Model
```python
model = torch.jit.load("model.pt")
model.eval()

X_t = torch.from_numpy(X)
y_t = torch.from_numpy(y)

with torch.no_grad():
    pred = model(X_t).squeeze()
```

### Step 3 — Writing submission.csv
```python
sub = pd.DataFrame({
    "subtaskID":  1,
    "datapointID": range(1000),
    "answer":      your_answer,
})
sub.to_csv("submission.csv", index=False)
```

## Submission File
Create `submission.csv` in the following format:

```text
subtaskID,datapointID,answer
1,0,1.234567
1,1,-0.567890
...
```

Where `answer` is the restored value for the corrupted column. The file must contain exactly 1000 rows.

## Evaluation Criteria

Your work will be evaluated based on the Mean Squared Error (MSE).

First, the MSE of your prediction is calculated:

$$ \text{MSE} = \frac{1}{1000} \sum_{i=0}^{999} (\hat{x}_i - x_i^*)^2 $$

Where \( \hat{x}\_i \) is your restored value, and \( x\_i^* \) is the original.

Then the score is scaled using the following formula:

$$ \text{Score} = \frac{\text{MSE}_{\text{baseline}} - \text{MSE}_{\text{yours}}}{\text{MSE}_{\text{baseline}} - \text{MSE}_{\text{best}}} \times 100 $$

And it is clipped within the \( [0, 100] \) range, where:

| Name                               | Value                | Description                 |
| :--------------------------------- | :------------------- | :-------------------------- |
| \( \text{MSE}_{\text{baseline}} \) | \( \approx 0.968 \)  | Leaving the column as zeros |
| \( \text{MSE}_{\text{best}} \)     | \( \approx 0.0045 \) | Best known solution         |

- Leaving the column as zeros → **0 Points**
- The more accurately you restore the original values → The higher the score (Maximum **100 Points**)

## Constraints

- **Model Integrity:** Modifying model weights is prohibited.
- **Submission:** It is mandatory to provide the corresponding code along with the answers.