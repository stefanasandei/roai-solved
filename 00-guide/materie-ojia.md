# Materia OJIA

## 1. Elemente de bază în Inteligența Artificială

Metrici de clasifiare:

- accuracy: cat la suta e corect
- recall: din toate cazurile reale de "pozitiv", cate ai gasit (spune cat de bine ai gasit toate cazurile pozitive)
- precision: din toate predictiile de "pozitiv", cate au fost corecte (spune cat de sigur esti ca un "pozitiv" prezis e intr-adevar pozitiv)
- F1 score: media armonica dintre precision si recall (deci echilibru intre cat de corecte si cat de complete sunt predictiile)
- confusion matrix: 2x2 cu TP, FN, FP, TN

Metrici pentru regresie:

- MAE: trateaza greselile proportional
- MSE: penalizeaza mai tare greselile mari
- MAPE: eroarea in procente
- log loss: binary cross entropy (2 clase); masoara cat de bine sunt calibrate scorurile de probabilitate pentru clasificare binara (also negative log likelihood)
- cross entropy loss: generalizare log loss pentru multiclass

Feature engineering:

- dummy encoding: one hot encoding, dar cu o coloana in minus

```py
dummy_cols = []
dummies_df = pd.get_dummies(df[dummy_cols], prefix=dummy_cols, drop_first=True)

df = df.drop(dummy_cols, axis=1)
df = pd.concat([df, dummies_df], axis=1)
```

- ordinal encoding:

```py
df["T Stage"] = df["T Stage"].map({"T1": 0, "T2": 1, "T3": 2, "T4": 3})
```

- datetime encoding: `.dt.*`

```py
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Is_AM"] = df["Timestamp"].dt.hour < 12
```

Procesarea datelor:

- filtrare: `df[df[col] > 3]`, `df[(corrs < 0.003) & (corrs > -0.003)]`
- `.nunique()`: cate valori unice sunt
- `.isna().sum()`: cate valori lipsesc per coloana
- `.value_counts()`: frecventa fiecarei valori per coloana
- `.sort_values()`: sortarea unei serii
- `.map(lambda x: x)`: transformarea valorilor dintr-o serie, o mapare - se apeleaza de pe o serie
- `.apply()`: aplicatii operatii (de sinteza) pe coloane - se apeleaza de pe un dataframe
- `.groupby()`: grupeaza datele dupa o serie de valori, necesar ca aceasta serie sa aiba duplicate pentru a avea sens agregarea

```py
>>> df = pd.DataFrame({'Animal': ['Falcon', 'Falcon',
...                               'Parrot', 'Parrot'],
...                    'Max Speed': [380., 370., 24., 26.]})
>>> df
   Animal  Max Speed
0  Falcon      380.0
1  Falcon      370.0
2  Parrot       24.0
3  Parrot       26.0
>>> df.groupby(['Animal']).mean()
        Max Speed
Animal
Falcon      375.0
Parrot       25.0
```

- `.agg({})`: pentru groupby, pentru a face agregari complexe mixte

```py
df.groupby("Base Flavor").agg({'Flavor Rating': ['mean', 'max', 'sum']})
```

- pivot tables: cand vrem date agregate dupa mai multe date categorice (deci groupby dar grupam dupa mai multe)

```py
df.pivot_table(index='Year', columns='Team', values='Height', aggfunc=['mean'])
# va rezulta un tabel cu Year pentru randuri, Team pentru coloane si valorile efective din tabel Height
```

Analiza datelor:

- `np.select(cond_list, choice_list)`: returneaza un array cu elemente din choice list, pe baza conditiilor din cond_list

```py
conditions = [
    test_df["GFR"] >= 90,
    (60 <= test_df["GFR"]) & (test_df["GFR"] < 90)
]

choices = ["Normal", "Mildly Decreased"]
subtask1 = np.select(conditions, choices)
```

- `np.quantile`: calculeaza quantile-ul

```py
q3 = np.quantile(series, 0.75)
```

## 2. Machine Learning

Linear Regression

- ideal pentru probleme simple de regresie cu relatii aproximativ liniare

Logistic Regression

- aplica functia sigmoid pentru a transforma rezultatul regresiei intr-o probabilitate
- parametrii:
  - `C` (inversul puterii de regularizare)
  - `penalty`: l1, l2, elasticnet, none
  - `solver`: liblinear lbfgs, sag, saga, newton-cg
  - `max_iter`

Naive Bayes

- familie de clasificatoare bazate pe Teorema lui Bayes si ipoteza de independenta intre caracteristici
- utilizat pentru clasificare text si probleme cu date discrete
- clasificare: `GaussianNB` si `MultinomialNB`

Bayes Regression

- regresie liniara utilizand rezultate ale Teoremei lui Bayes

K-nearest neighbors

- atribuie eticheta sau valoarea medie in functie de cei mai apropiat k vecini in spatiul caracteristicilor
- parametrii:
  - `n_neighbors`

K-means

- algoritm de clustering, grupeaza datele in k clustere, minimizand intertia
- parametrii:
  - `n_clusters`

Decision trees

- imparte recursiv spatiul caracteristicilor pe baza unor criterii (gini, entropy, etc.) pana la frunze omogene
- parametrii:
  - `max_depth`
  - `n_estimators`
  - `criterion`: gini, entropy, log_loss
  - `min_samples_leaf`
  - `min_samples_split`

Random forest

- o grupare de arbori de decizie, voteaza (clasificare) sau fac media (regresie)
- parametrii:
  - `n_estimators`
  - `max_depth`

XGBoost

- bilbioteca optimizata pentru gradient boosting, se construiesc secventiari arbori de decizie minimizand loss-ul si aplicand penalizari pentru regularizare
- parametrii:
  - `learning_rate`
  - `max_depth`
  - `n_estimators`
  - `subsample`
  - `colsample_bytree`

## 3. Natural Language Processing

Procesare text:

- `.str.lower()` si `.str.upper()`
- `.str.len()`
- `.str.strip()`: eliminarea spatiului gol de la inceput si final
- `.str.replace("ceva", "altceva")`
- `.str.split("_").str[1]`: al doilea element dupa impartirea in array dupa "\_"
- `.str.removeprefix("str_")`
- pentru a obtine cuvintele unice dintr-o serie:

```py
result = set()
df['text'].str.lower().str.split().apply(result.update)
```

Bag of words:

- se asigneaza un id numar intreg fiecarui cuvant din orice document
- pentru fiecare document `#i` si stocheaza de cate ori a aparut cuvantul `w` si stocheaza in `X[i, j]`, unde `j` este indexul cuvantului `w` din dictionar

- deoarece in matrice vor fi extrem de multe valori de 0, se foloste o structura de date sparse

```py
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(twenty_train.data)
```

Term Frequency-Inverse Document Frequency (TF-IDF):

- exista problema cand numarul de aparitii afecteaza performanta modelului (documente mai lungi vor avea un count mai mare, dar subiectul poate fi acelasi); de aceea vom imparti numarul de aparitii al fiecarui cuvant la numarul total de cuvinte din document

```py
from sklearn.feature_extraction.text import TfidfTransformer

vectorizer = TfidfTransformer(smooth_idf=False)

X_train = vectorizer.fit_transform(data_train.data)

feature_names = vectorizer.get_feature_names_out()
```

## 4. Computer Vision

Procesare imagini:

- fiecare valoare vine in coloana ei

```py
df["pixels"] = df["pixels"].apply(eval)

pixel_cols = pd.DataFrame(df["pixels"].tolist())
pixel_cols.columns = [f"pixel_{i}" for i in range(4096)]

df = pd.concat([df.drop(["pixels"], axis=1), pixel_cols], axis=1)
```

- se normalizeaza imaginile:

```py
X = df_train
global_mean_vector = X.mean(axis=0) # se foloseste si pt train si pt test

X_centered = X - global_mean_vector
```

Augmentarea datelor:

- cropping
- rotation
- flipping
- filters (sharpen, etc.)
- color change (saturation, contrast)
- affine transformation (translation, sheer, scale)
- noise

Mai multe detalii in ghidul pentru librarii.
