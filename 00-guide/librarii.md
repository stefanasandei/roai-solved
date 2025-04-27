# numpy

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

# pandas

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

# matplotlib

```py
import matplotlib.pyplot as plt

plt.figure(figsize=(19, 2))
plt.xticks(rotation=90)

plt.plot(X, y)
plt.show()
```

# scikit-learn

Data loading:

```py
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Modelare:

```py
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy_score(y_test, y_pred)
```

Encoding (label, ordinal, onehot, etc.):

```py
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
y = enc.fit_transform(y)
```

Polynomial features:

```py
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(3)
poly.fit_transform(X)
```

Finetuning:

```py
from sklearn.grid_search import GridSearchCV
params = {"n_neighbors": np.arange(1,3), "metric": ["euclidean", "cityblock"]}

grid = GridSearchCV(estimator=knn, param_grid=params)
grid.fit(X_train, y_train)

print(grid.best_estimator_, grid.best_score_)
```

# opencv

```py
import cv2

image = cv2.imread('path.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

Edge detection:

```py
edges = cv2.Canny(gray_image, 50, 150)
```

# nltk

```py
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

tokens = word_tokenize(text.lower())

# remove stop words
filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

# lemmatize the tokens
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

# join the tokens back into a string
processed_text = ' '.join(lemmatized_tokens)
```

Sentiment analysis:

```py
analyzer = SentimentIntensityAnalyzer()

scores = analyzer.polarity_scores(text)
sentiment = 1 if scores['pos'] > 0 else 0
```

# pillow

TODO

# scikit-image

TODO

# spacy

```py
import spacy
# Load the installed model "en_core_web_sm"
nlp = spacy.load("en_core_web_sm")

doc = nlp("This is a text")
[token.text for token in doc]
# ['This', 'is', 'a', 'text']
```

Part-of-speech tags:

```py
doc = nlp("This is a text.")
# Coarse-grained part-of-speech tags
[token.pos_ for token in doc]
# ['DET', 'VERB', 'DET', 'NOUN', 'PUNCT']
```

Word vectors:

```py
# Vector as a numpy array
doc = nlp("I like cats")
# The L2 norm of the token's vector
doc[2].vector
doc[2].vector_norm
```

# scipy

TODO

# seaborn

TODO

# xgboost

TODO

# optuna

TODO
