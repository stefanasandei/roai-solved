# Operation "Vano Papa's Heritage"

https://judge.nitro-ai.org/competitions/gaia/ai-league-ii/2/view

## History

The legendary Georgian programmer, **Vano Papa**, hid his Bitcoin passwords in 200 Sudoku boards. The problem is that Vano Papa hated storing passwords on a computer; consequently, he wrote everything by hand on paper. You have been handed his "digital chest," where hundreds of handwritten Sudokus are scattered.

Your mission is to create an algorithm that can distinguish Papa's scribbles and determine which Sudoku is correctly filled and which is not.

## Task Essence

You must create a system that determines whether the Sudoku provided in a photo is valid and correctly filled or not.

## Sudoku Validity Rules

A board is considered correctly filled (`answer = 1`) if:

- Every **row** contains digits from 1 to 9 exactly once.
- Every **column** contains digits from 1 to 9 exactly once.
- Every **3×3 square** contains digits from 1 to 9 exactly once.

## Provided Materials

### 1. MNIST Database
The standard MNIST handwritten digit database — 60,000 training and 10,000 testing images (digits 0–9, each 28×28 pixels, grayscale). Located in `mnist/`.

### 2. Training Boards — 20 Images
Located in `train_boards/`. Each board is a **252×252 pixel grayscale PNG** image containing 81 cells in a 9×9 grid. Each cell is exactly 28×28 pixels.

**Board Labels** — `train_labels.csv`:

```text
subtaskID,datapointID,answer
1,0,1
1,1,0
...
```

### 3. Test Boards — 200 Images
Located in `test_boards/`. The image format is the same as the training boards. **Answers are not provided.** Your program must decide the validity itself.

## Submission Format

Create a file named `submission.csv`:

```text
subtaskID,datapointID,answer
1,0,1
1,1,0
1,2,1
...
```

Where `answer` is `1` for a valid Sudoku and `0` for an invalid one. The file must contain exactly 200 rows (one for each test board).

## Data Reading

### Step 1 — Loading Training Boards

```python
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_labels = pd.read_csv("train_labels.csv")
# Columns: subtaskID, datapointID, answer (1=valid, 0=invalid)
print(train_labels.head())
```

### Step 2 — Loading MNIST

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

mnist_train = datasets.MNIST("mnist", train=True, download=False, transform=transform)
mnist_test = datasets.MNIST("mnist", train=False, download=False, transform=transform)

print(f"MNIST train: {len(mnist_train)} images")
print(f"MNIST test:  {len(mnist_test)} images")
```

### Step 3 — Loading Test Boards

```python
test_boards = []
for i in range(200):
    img = Image.open(f"test_boards/{i}.png").convert("L")
    test_boards.append(np.array(img))

print(f"Loaded {len(test_boards)} test boards, size: {test_boards[0].shape}")
```

### Step 4 — Writing submission.csv

```python
predictions = [0] * 200  # ← Replace with your predictions (0 or 1)

sub = pd.DataFrame({
    "subtaskID":  1,
    "datapointID": range(200),
    "answer":      predictions,
})
sub.to_csv("submission.csv", index=False)
```

## Constraints

- Code is mandatory — submissions without code will not be considered.
- Your solution must be automated — hardcoded answers will result in disqualification.

## Evaluation

The score is calculated based on the classification accuracy of the 200 test boards.

First, accuracy is calculated:

$$ \text{Accuracy} = \frac{\text{Number of correctly classified boards}}{200} $$

Then the score is scaled using the following formula:

$$ \text{Score} = \frac{\text{Accuracy} - 0.50}{0.995 - 0.50} \times 100 $$

And it is clipped to the [0, 100] range, where:

| Name  | Value | Description                              |
| :---- | :---- | :--------------------------------------- |
| 0.50  | 50%   | Trivial solution (all True or all False) |
| 0.995 | 99.5% | Best known solution                      |

- Guessing all boards the same (50% accuracy) → **0 points**
- Reaching the best known solution (99.5% accuracy) → **100 points**