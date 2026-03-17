# Regional Phase 2025

Link: https://judge.nitro-ai.org/competitions/roai-2025/ojia

## 🚚 Smart Cargo: 100p

- subtask 1: basic dataframe filtering
- subtask 2: dummy encoding for weather + linear regression

## 🤾♂️ Caloric Consumption: 100p

- subtask 1: len on the dataset
- subtask 2: len + filtering
- subtask 3: .mean()
- subtask 4: length + filtering
- subtask 5:
  - dummy encoding for gender (where Gender_male is crucial)
  - degree 3 polynomial features (features resulting from multiplication with Gender_male, acting as a mask, will be important)
  - ridge regression, for regularization (avoiding overfitting)
- subtask 6: the same model is reused, but with predictions on the filtered dataset
- 