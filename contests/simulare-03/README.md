# Simulare OJIA 3

Link: https://judge.nitro-ai.org/roai-2025/simulare-ojia-3

## Problema Admiterii la un liceu de elită: 98p

- subtask 1: scadere
- subtask 2: subtask gresit
- subtask 3: drop la gen si judet + logistic regression

Observatie: S-au obtinut doar 58/60 la subtask-ul 3.

## Predicția scorului la examen: 100p

- subtask 1: diferenta
- subtask 2: filtrare + map
- subtask 3 .apply
- subtask 4 .value_counts
- subtask 5:
  - drop la gen
  - se inlocuiesc valorile missing cu mode-ul
  - se face ordinal mapping
  - dummy encoding pe restul
  - model ridge regression
