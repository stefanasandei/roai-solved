# NEOAI 2025

https://www.kaggle.com/competitions/neoai-2025/overview

## Status

Percentage formula: `Norm_Score = (Submission_Score - Baseline) / (Max_Score - Baseline) * 100`

| Task | Score | Baseline | Max score | Percentage | Type |
| ---- | ----- | -------- | --------- | ---------- | ---- |
| 1    | 2.95  | 3.14     | 2.24      | 21%        | ML   |
| 2    | 0.72  | 0.53     | 0.78      | 75%        | CV   |
| 3    | -     |          |           |            | NLP  |
| 4    | 0.04  | 0.02     | 0.11      | 22%        | ML   |
| 5    | 0.42  | 0.36     | 0.78      | 14%        | NLP  |
| 6    |       |          |           |            | CV   |
| 7    |       |          |           |            | CV   |
| 8    |       | 0.29     | 0.58      |            | NLP  |

## Explanations

### Task 1: [Tracy tables](https://www.kaggle.com/code/timriggins/basel1ne-tricy-table-data)

Summary: given a table with plenty unknown features, do feature engineering to increase the score of a lightgbm model

### Task 2: [Underfitting CV](https://www.kaggle.com/code/timriggins/baseline-cv-underfitting)

Summary: given a vision transformer trained on 90 classes and images corresponding to 10 other classes, fine-tune the ViT on the new classes, without degrading performance too much on the 90 classes. There is no access to images from 90 classes. 

### Task 3: [Evading AI-Generated Text Detection](https://www.kaggle.com/code/egorgij21/baseline)

Summary: given a gemma2 2b model, change its outputs to be more human-like, according to a given dataset, do not finetune

Note: vram intensive (doesn't works on google colab free tier); used gpt2-small instead of gemma2 2b

### Task 4: [Cluster images](https://www.kaggle.com/code/timriggins/basel1ne-cluster-1mages)

Summary: given arrays of heavily augmented images, which come from original 32 images, cluster the images into 32 clusters

### Task 5: [Broken BERT](https://www.kaggle.com/code/ilseyaralimova/broken-bert-baseline)

Summary: given a BERT model with broken embeddings (some tokens have embeddings fully null), fix the embeddings without fine-tune.

### Task 6: [The Hogspell Challenge](https://www.kaggle.com/code/lenjjiv/en-hogspell-baseline-solution)

Summary: fine-tune stable diffusion 1.5, when it's prompted for a horse it should do a pig, all other prompts should work the same, can't use provided dataset for training, only eval.

Note: vram intensive (works on google colab free tier)

### Task 7: [Cuties Segmentation](https://www.kaggle.com/code/tatianagaintseva/baseline-eng)

Summary: given a pretrained CLIP, produce binary segmentation masks for cats & dogs of different breeds, training on validation data is allowed

Note: vram intensive (works on google colab free tier)

### Task 8: [Intent Detection and Slot Filling](https://www.kaggle.com/code/ilseyaralimova/baseline-for-nlp-task)

Summary: train a joint BERT model for intent classification and slot filling, there are train/validation/test datasets.
