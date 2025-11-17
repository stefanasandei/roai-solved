# Cram School Easter Round

Link: https://judge.nitro-ai.org/cram-school/cram-school-practice

## Cybersecurity AI Challenge: 100p

### subtask 1

- datetime features: `pd.to_datetime`
- dupa care facem o noua coloana bool, cu valoarea `.dt.hour < 12` de la coloana timestamp
- iar pentru raspuns, mapam true la "AM" si false la "PM"

### subtask 2

- se inlocuiesc valorile missing cu media
- se foloseste un model random forest cu 300 estimatori

## Brain Anomaly Detection: 100p

### subtask 1

- trebuie convertit array-ul string de pixeli intr-un array: `df["pixels"] = df["pixels"].apply(eval)`, este evaluata expresia
- dupa care facem un dataframe care sa aiba 4096 de coloane cu pixeli, cate un float per coloana (astfel putem avea un feature vector X care sa fie intreaga imagine, putem lucra cu modele traditionale sklearn)
- se asigneaza la fiecare coloana un nume `pixel_{i}`

- vectorul `X` va fi dataframe-ul rezultat, se calculeaza media cu `.mean()`
- vectorul centrat va fi `X - X.mean(axis=0)`

Obs: aceasta este o preprocesare necesara a features-urilor, se face si pentru dataset-ul de antrenare si pentru cel de testare

### subtask 2

- dupa ce s-a facut preprocesarea necesara, este suficient un model `MLPClassifier` sau `SVC`
- scrierea solutiilor in csv este mai complexa, trebuie utilizat `np.array` si este corect aparent ca raspunsul sa fie prescurtat cu `...`
- ideea este ca construim randurile manual intr-un for, rand cu rand, apoi adunam randurile pentru cele doua subtask-uri si scriem numele coloanelor
