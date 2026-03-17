# Selection Camp 2025

Selection Round 1 (CV): https://judge.nitro-ai.org/roai-2025/lot-baraj-1/

Selection Round 2 (NLP): https://judge.nitro-ai.org/roai-2025/lot-baraj-2/

Selection Round 3 (Theory): https://olimpiada.nitro-ai.org/2025/lot/subiecte-teorie.pdf

During the contests, the datasets could be downloaded from http://roai-docs.olimpiada-ai.ro/. Just in case, I saved the zips in a Google Drive archive so they are not lost: https://drive.google.com/drive/folders/1CscTLaJADKRQASn3s7UeTFpSbrkNt7TQ?usp=sharing

## Status

| Task        | Score   | Type |
| ----------- | ------- | ---- |
| Hotspot     | **98**  | CV   |
| Angry Birds | **-**   | CV   |
| Toxic       | **100** | NLP  |
| Skeletons   | **100** | NLP  |

## Explanations

### Task 1: [HotSpot](https://judge.nitro-ai.org/competitions/roai-2025/lot-baraj-1/1/view)

Summary: 4 lists of images are provided, consisting of black backgrounds + colored geometric shapes; noise elements are gradually added. Binary segmentation masks must be calculated only for the geometric shapes.

Solution 96/100: No model is used; only image preprocessing with OpenCV. For each image, we apply grayscale, median blur (to handle noise + stripes), then thresholding (using a different value for subtask 4). Each image is processed, then RLEs are created and the solution is written.

Solution 100/100: Explanation coming soon!

### Task 2: [Angry Birds](https://judge.nitro-ai.org/competitions/roai-2025/lot-baraj-1/2/view)

Summary: A ResNet50 is provided and must be fine-tuned to classify 2 types of images: water bird or land bird. Elements were added to the dataset to increase difficulty: a red square in a random position was added to land bird images (only in the training set). There are images of water birds on land/water backgrounds and land birds on water/land backgrounds. The training dataset is imbalanced regarding these bird/background types. Evaluation is based on the lowest accuracy among the 4 classes: water/land bird on water/land background.

Explanation coming soon!

### Task 3: [How Toxic Are You Online? ](https://judge.nitro-ai.org/competitions/roai-2025/lot-baraj-2/1/view)

Summary: A dataset with online comments is provided; each comment must be classified as toxic/severe_toxic/obscene/insult.

Solution 100/100 (F1 of 0.6991):

- The text is preprocessed: unnecessary characters (punctuation, numbers, etc.) are removed, web addresses are replaced with `WEB`, contractions are expanded, and finally, tokens are lemmatized.
- A word-level **tfidf** with n-grams of (1, 2) is applied.
- We train 4 different **catboost** models for each label, each with specifically fine-tuned parameters (sklearn's `MultiOutputClassifier` could be used to reduce code, but it offers less flexibility).
- For each trained model, we **calculate the optimal threshold** using the precision-recall curve: the predicted probability is between 0 and 1; performance can be improved depending on the threshold value used to determine a 0 or 1 output—this is a key observation in the solution.

Solution 100/100 (F1 of 0.734313): fine-tuning BERT; I believe a much higher score can be achieved with a larger sequence length (currently 128) and more epochs (currently 8, but improvement was still visible in the final epoch).

### Task 4: [Skeletons Don’t Lie: Can AI Decode Your Moves?](https://judge.nitro-ai.org/competitions/roai-2025/lot-baraj-2/2/view)

Summary: Camera position and the executed action must be classified in a series of videos. A video is represented by a sequence of XYZ coordinates for human joints (25 x 3 coordinates per frame), with a variable number of frames per video. The [provided PDF](./nlp/skeletons/explicatie.pdf) explains the coordinate system.

Explanation coming soon!

<!-- 100p sub: https://judge.nitro-ai.org/competitions/roai-2025/lot-baraj-2/2/submissions/25bdda25-576f-4180-8aae-43a77249771f -->