# National Phase 2025

Link: https://judge.nitro-ai.org/competitions/roai-2025/onia

## 🧔 Human vs AI 🤖: 100p

### Subtask 1 - classification

- minimal text cleaning (lowercase + no leading/trailing spaces); this is necessary because "noise" characters can be strong indicators of whether text is AI-generated (e.g., ChatGPT consistently uses correct punctuation, frequent emojis, specific unicode characters, etc.)
- a TF-IDF vectorizer is used on _characters_ (`analyzer='char_wb'`); this is relevant as we want distinct features for punctuation, unicode characters, etc.
- a `LinearSVC` is applied and yields the maximum score

### Subtask 2 - clustering

- traditional text cleaning is applied (removing stop words and punctuation, stemming)
- a standard TF-IDF is used
- clustering is performed using K-Means on the TF-IDF features
- to visualize the clusters and ensure the clustering makes sense, we use `TSNE` and create a scatter plot
- after verifying that the clusters look correct, we display the most relevant features from each cluster and manually assign names

## Byzantine Notation: 100p

- elementary transformations are applied to the training dataset (random rotation + color jitter)
- a basic CNN is trained (2 conv layers, max pooling, dropout, and finally 2 fully connected layers)
- the training phase proceeds smoothly

the interesting part occurs during submission creation:

- since there are multiple neumes per image, they must be selected (our model works _only_ with 48x48 grayscale images containing a single neume)
- using OpenCV, we first binarize the image—converting it to black and white (removing background)—using thresholding and blurring
- once the image is clean, we use the `findContours` function in OpenCV to identify distinct objects
- for each contour, we use the bounding box to check if it is noise; if legitimate, it is added to the list of patches
- subsequently, each patch image is resized to 48x48 and processed through the model
