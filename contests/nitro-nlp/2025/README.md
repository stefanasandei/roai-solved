# Nitro NLP 2025

Link: https://judge.nitro-ai.org/nitro-2025/nitro-nlp

## Mickey și Donald: 77p

- s-au folosit modele diferite pentru cele doua subtask-uri
- preprocesare text:
  - totul in litere mici
  - inlocuirea `$NE$` (devenit `$ne$`) cu un token specific unic (am ales `entitate_necunoscuta`), asta e deoarece ulterior vectorizer-ul va ignora `$`, astfel pastram structura
  - inlocuim â din a cu â din i (si variantele mari)
- se foloseste un Tfidf vectorizer cu range ngram de (1,3) si selectam structurile care apar in minim 2 documente si maxim 90%
- scadem cu 1 toate clasele pentru antrenare, cand scriem submisia adaugam 1 la predictii

subtask 1:

- mai intai facem un model MultinomialNB cu alpha 0.001 (s-au incercat manual valori - finetuning) - folosim si un model LinearSVC cu C de 50 (am incercat si mai mare, nu aduce improvements notabile)
- in final facem un stacking ensemble din cele doua, notabil este ca urmarim nu doar un scor cat mai mare ci si o tendinta minima de overfitting (urmarire diferenta scor CV cu scorul normal)

subtask 2:

- folosind ce am aflat de la subtask 1, folosim un singur model MultinomialNB

## Find the Ducks: 90p

- rezolvarea este in afara programei ONIA
- se foloseste un model preantrenat `resnet50` pentru duck detection
- pentru bounding box se foloseste un model `yolo11m`
- pentru pixel count se foloseste un model gradient boosting regressor

```
19 / 20 Accuracy: 0.958
27 / 35 Root Mean Squared Error: 0.793082
34 / 45 Mean Squared Error: 0.989532
Total: 90 / 100
```

Se recomanda rularea notebook-ului pe bucati, mai intai se antreneaza modelul de duck detection, se scrie modelul `.pt`, apoi se da restart, se antreneaza modelul de bounding box si apoi submisia se face cu cele doua modele citite din fisierele `.pt`.
