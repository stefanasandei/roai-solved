# Simulare OJIA 1

Link: https://judge.nitro-ai.org/roai-2025/simulare-ojia

## Livrare pachete: 100p

- Subtask 1: np.mean
- Subtask 2: np.std(traffic_lvl, ddof=1) sau traffic_lvl.std() (ideea e ca termenul din fata este 1/(n-1), nu 1/n)
- Subtask 3: un logistic regression sau random forest simplu

## Credit score: 100p

- Subtask 1: len
- Subtask 2: filtrare basic
- Subtask 3: .nunique()
- Subtask 4: filtrare cu .str.endswith()
- Subtask 5:
  - random forest (cu n_estimators 250) pentru 65/80
  - pentru 100, e necesar un basic feature engineering (.apply, .get_dummies)
