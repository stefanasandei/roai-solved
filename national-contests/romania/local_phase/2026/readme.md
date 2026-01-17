# Local Phase 2026

Link: https://platform.olimpiada-ai.ro/en/competitions/7

## "Analysis and Classification of Parkinson's Risks and Symptoms": 100p

**TLDR**
The goal is to analyze clinical data to predict whether patients suffer from Parkinson's disease (binary classification) while also calculating specific risk scores. The dataset includes demographic details, lifestyle factors, and specific medical indicators like tremor presence and cholesterol levels.

### Subtask 1
I calculated the cardiovascular risk score by implementing the provided boolean logic directly on the dataframe. I summed the binary outcomes of whether the patient has `Hypertension`, has `Diabetes`, and has a `BMI` greater than 30.

### Subtask 2
Similarly to the first subtask, I computed the lifestyle index by summing three conditions: whether `Smoking` is true, if `AlcoholConsumption` is strictly greater than 2, and if `PhysicalActivity` is strictly less than 1 hour per week.

### Subtask 3
To handle the classification, I first cleaned the data by dropping non-predictive columns such as `PatientID` and `Ethnicity`. I split the data to validate different models, testing Logistic Regression and Random Forest, but settled on a `CatBoostClassifier` because it yielded the highest AUC. I trained the model with 1000 iterations and a learning rate of 1e-3 on the full training set (utilizing both the provided train data and the calculated risk scores) to generate the final 0/1 predictions for the test set.

## "Famous Paintings": 100p

**TLDR**
This task involves analyzing a dataset of paintings to determine their authenticity based on strict rules, grouping them by their anonymous painter, and predicting their market price. The data provides attributes such as canvas size, stroke density, complexity, and brush type.

### Subtask 1
I implemented a vectorized scoring system to calculate the Artistic Authenticity Score (AAS). I added points for specific criteria like `stroke_density` $> 0.7$ and `complexity` $> 0.65$, and subtracted points if `contrast` or `brightness` fell outside specific ranges. I assigned the label "Autentic" if the final calculated array was $\ge 5$, and "Incert" otherwise.

### Subtask 2
To identify the five painters, I isolated stylistic features such as `painter_style_score`, `fake_style_score`, and `complexity`. I standardized these features using `StandardScaler` to ensure equal weighting. Since the problem statement confirmed there were 5 painters, I used `KMeans` clustering with $k=5$ on the combined train and test sets to assign a group ID to each painting.

### Subtask 3
For price prediction, I performed significant feature engineering, splitting `canvas_size` to calculate `canvas_area` and aspect ratio, and one-hot encoding categorical variables. A crucial detail in my approach was avoiding data leakage: I trained a `KMeans` model solely on the training set's stylistic features and used it to predict cluster labels for both train and test sets. I added this cluster label as a feature (`cluster_feat`) to the regression data, allowing the model to account for the painter's "brand" value. Finally, I used `Ridge` regression on the scaled features to predict the `target_price`.

## "Public Transportation": 100p

**TLDR**
You are provided with GPS telemetry data spanning three days for public transport vehicles in multiple cities. The objective is to determine fleet statistics, group vehicles by city based on their location history, and identify depot locations for a specific vehicle type using nighttime coordinates.

### Subtask 1
I calculated the number of unique vehicles and vehicle types by applying the `nunique()` function to the `id` and `vehicle_type` columns, respectively.

### Subtask 2
To associate vehicles with cities, I first reduced the noise in the GPS data by calculating the median `latitude` and `longitude` for every unique `id`. Using these aggregated coordinates, I applied `DBSCAN`, a density-based clustering algorithm, with an epsilon of 0.5. This effectively grouped the vehicles into spatially distinct clusters corresponding to the different cities.

### Subtask 3
To find the depots, I filtered the dataset to include only records where `vehicle_type` was 10. Since vehicles return to depots at night, I further filtered for timestamps where the hour was either 23 or strictly less than 3 (23:00–03:00). I applied `KMeans` clustering with $k=3$ on these specific nighttime coordinates. The centroids of these clusters represented the depot locations, which I then sorted by latitude as required.