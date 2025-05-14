# Cram School PreONIA 2025

Link: https://judge.nitro-ai.org/cram-school/cram-school-preonia-2025

explicatii todo

## The Reality Filter: 80p

- subtask 1: inmulitere
- subtask 2: cu filtrare + .value_counts() gaseam numerele, apoi faceam raportul (observatie: acest subtask este foarte relevant caci aflam ca datasetul de inbalanced)
- subtask 3:
  - folosind sampling, aducem la egalitate numarul de date cu cele doua labeluri (am rezolvat inbalanced-ul)
  - augmentarea imaginilor: am marit datasetul de 4 ori (mirror-ing, rotate & color jitter)
  - imaginile au fost citite ca array numpy, dupa care procesate in range-ul [0.0, 1.0]
  - in final s-a folosit un SVC cu C=10 pentru o acuratete de ~0.675213 in contest

Observatie: editorialul oficial propune un CNN "simplu" pentru maxim de puncte, in practica nu functioneaza fiind omise detalii cruciale pentru implementare de maxim.

## The Quadrilingual Land of Lonpestia: 100p

- se impartea in doua dataframeuri: unul cu ambele texte si unul doar cu textA
- problema e de unsupervised learning, de clustering

### subtask 1

- folosin un tfidf vectorizer cu analyzer `char_wb` si ngram range de (2, 4)
- concatenam cele doua coloane de texte si antrenam vectorizer-ul pe corpusul complet
- transformam ulterior cele doua coloane de text individual
- se calculeaza similaritatea cosinus si facem histograma
- cautam ochiometric un threshold (0.3 merge)
- facem filtrarea si toate cu similaritatea >= 0.3, sunt True (aceeasi limba)

### subtask 3

- curatam textul (scoatem spatiile, litere mici, etc.)
- folosim un tfidf vectorizer similar, doar pe textA
- antrenam un Kmeans in 4 clustere
- _important_: afisam primele 10 cele mai relevante features din fiecare cluster, astfel putem determina ce limba corespunde carui cluster (cuvintele seamana cu numele limbii, aproximativ)
- folosim tSNE pentru a vizualiza cele 4 clustere, sanity check
- in final, mapam de la indice de cluster la numele limbii
