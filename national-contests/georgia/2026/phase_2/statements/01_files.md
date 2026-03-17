# Giorgi's Files

https://judge.nitro-ai.org/competitions/gaia/ai-league-ii/1/view

Giorgi, a successful data scientist, spent years collecting and labeling important data. Giorgi knew his data was highly sought after, so he created his own cipher to encrypt the entire archive of training data.

Valeri, your colleague, obtained Giorgi's files. He discovered that:

*   Each datapoint is a **vector of length 1024**, where each value is a single byte (a number from 0 to 255).
*   Each datapoint has a **target variable** — an integer from 0 to 9.

A large part of Giorgi's archive is encrypted, but through great effort, Valeri managed to obtain three files, including one small document:

> ### EFTA02427738.txt
>
> "Everyone is trying to steal my files. Because of this, I created a cipher that transforms raw data through several linear transformations without reducing the size.
>
> The efficiency of its operation can be seen in the `public_pairs.npz` file."

It is noteworthy that Valeri **does not know** what the raw data actually represents.

**Your goal** is to help Valeri: figure out the working principle of the cipher, decrypt the encrypted data, and based on it, create a classifier that will label the real test set.

## Provided Materials

### 1. `public_pairs.npz` — Cipher Sample
Contains **one pair** of a real vector and its corresponding encrypted vector:

*   `x_real` — real vector, shape: `(1, 1024)`, dtype: `uint8`
*   `x_encrypted` — encrypted vector, shape: `(1, 1024)`, dtype: `float32`
*   `datapointID` — unique identifier, shape: `(1,)`

### 2. `public_encrypted_train.npz` — Encrypted Training Set
Contains ~48,000 encrypted datapoints along with labels:

*   `x_encrypted` — encrypted vectors, shape: `(N, 1024)`, dtype: `float32`
*   `y` — class labels (0–9), shape: `(N,)`, dtype: `int64`
*   `datapointID` — unique identifiers, shape: `(N,)`

### 3. `public_query_real.npz` — Real Test Set
Contains 2000 real (unencrypted) vectors **without labels**:

*   `x_real` — real vectors, shape: `(2000, 1024)`, dtype: `uint8`
*   `datapointID` — unique identifiers, shape: `(2000,)`

### 4. `sample_output.csv` — Submission Sample
A sample file showing the format in which you should upload your answers.

## Reading Data

### Loading Files

```python
import numpy as np

pairs = np.load("public_pairs.npz")
enc_train = np.load("public_encrypted_train.npz")
query_real = np.load("public_query_real.npz")

print(f"Pair: real {pairs['x_real'].shape}, encrypted {pairs['x_encrypted'].shape}")
print(f"Encrypted train: {enc_train['x_encrypted'].shape}, labels: {enc_train['y'].shape}")
print(f"Real query: {query_real['x_real'].shape}")
```

### Writing submission.csv

```python
import pandas as pd

query_ids = query_real['datapointID']
predictions = [0] * 2000  # ← Replace with your prediction (from 0 to 9)

sub = pd.DataFrame({
    "subtaskID":  1,
    "datapointID": query_ids,
    "answer":      predictions,
})
sub.to_csv("submission.csv", index=False)
```

## Submission Format

Create a file `submission.csv`:

```text
subtaskID,datapointID,answer
1,1,3
1,2,7
1,3,0
...
```

Where:

*   `subtaskID` is always `1`
*   `datapointID` corresponds to the identifiers given in `public_query_real.npz`
*   `answer` is your prediction: an integer **from 0 to 9**

The file must contain exactly **2000 rows** (one per test datapoint).

## Constraints

*   Code is mandatory — submission without code will not be counted.
*   Your solution must be automated — hardcoded answers will lead to disqualification.
*   The use of pre-trained models is prohibited.

## Evaluation

The score is calculated based on the **Weighted F1** metric on the 2000 test datapoints.

Weighted F1 calculates the F1 for each class separately and then takes their weighted average (weight = class frequency):

$$ \text{Weighted F1} = \sum_{c=0}^{9} \frac{n_c}{N} \cdot F_1^{(c)} $$

Then the score is scaled using the following formula:

$$ \text{Score} = \frac{\text{Weighted F1} - F_{\text{sample}}}{F_{\text{best}} - F_{\text{sample}}} \times 100 $$

And is clipped to the range \( [0, 100] \), where:

| Name                    | Value | Description                    |
| :---------------------- | :---- | :----------------------------- |
| \( F_{\text{sample}} \) | ~0.10 | Sample result (random answers) |
| \( F_{\text{best}} \)   | ~0.89 | Best known solution            |

*   Random answers (~0.10 Weighted F1) → **0 points**
*   Best known solution (~0.89 Weighted F1) → **100 points**