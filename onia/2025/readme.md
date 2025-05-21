# ONIOA 2025

Link: https://judge.nitro-ai.org/roai-2025/onia

## 🧔 Om vs AI 🤖: 100p

### Subtask 1 - clasificare

- curatare minima a textului (litere mici + fara spatii la inceput si final), acest lucru este necesar deoarce caractere de "noise" pot fi un indicator bun daca un text este generat de AI sau nu (exemplu: chatgpt scrie mereu corect cu toate semnele de punctuatie, foloseste multe emoji, mai foloseste si caractere unicode specifice, etc.)
- se foloseste un tf idf vectorizer, pe _caractere_ (`analyzer='char_wb'`), acest lucru este relevant deoarece vrem features distincte pentru semne de punctuatie, caractere unicode, etc.
- se aplica un `LinearSVC` si iesa de maxim

### Subtask 2 - clustering

- acum vom aplica o curatare traditionala a textului (fara stop words si punctuatie, stemming)
- se foloseste un tfidf banal
- facem clustering, cu kmeans, de pe feature-urile de la tfidf
- pentru a vizualiza clusterele, sa ne asiguram ca are sens clusteringul, folosim `TSNE` si facem un scatter plot
- dupa ce am vazut ca arata bine clusterele, afisam cele mai relevante features din fiecare cluster si asignam la mana numele

## 𝁘 Notație Bizantină 𝁑: 77p

- se aplica transformari elementare pe datasetul de antrenare (random rotation + color jitter)
- antrenam un cnn elementar (2 layere conv, max pooling, dropout si la final 2 layere fully connected)
- decurge smooth partea de antrenare

partea interesanta vine la crearea submisiei:

- avand mai multe neume per imagine, trebuie cumva sa le selectam (modelul nostru lucreaza _doar_ cu imagini grayscale 48x48 cu o singura neuma)
- folosind opencv, mai intai binarizam imaginea - adica o facem doar in alb si negru (remove background), asta se face cu un thresholding si un blurring
- dupa ce avem imaginea curata, folosim functia `findContours` din opencv pentru a gasi obiectele distincte
- pentru fiecare contur, folosind bounding boxul verificam daca e noise sau nu, iar daca e legit il punem in lista de patches
- ulterior fiecare imagine in patches va fi resized la 48x48 si va fi rulata prin model

Observatie: In contest, s-a luat si 99p la problema de CV, deci teoretic este posibil, dar nu consider ca merita timpul aditional de lucru, cand rezolvarea este in principal la fel (ajustare hyperparametrii, seed, etc.).
