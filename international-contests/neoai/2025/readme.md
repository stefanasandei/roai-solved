# NEOAI 2025

https://www.kaggle.com/competitions/neoai-2025/overview

## Status

Percentage formula: `Norm_Score = (Submission_Score - Baseline) / (Max_Score - Baseline) * 100`

| Task | Score    | Baseline | Best Score | Percentage | Type |
| ---- | -------- | -------- | ---------- | ---------- | ---- |
| 1    | **2.95** | 3.14     | 2.24       | **21%**    | ML   |
| 2    | **0.72** | 0.53     | 0.78       | **75%**    | CV   |
| 3    | **-**    | 0.07     | 0.32       | **-**      | NLP  |
| 4    | **0.08** | 0.02     | 0.11       | **66%**    | ML   |
| 5    | **0.42** | 0.36     | 0.44       | **75%**    | NLP  |
| 6    | **0.93** | 0.40     | 0.99       | **90%**    | CV   |
| 7    | **0.78** | 0.36     | 0.73       | **100%**   | CV   |
| 8    |          | 0.29     | 0.58       |            | NLP  |

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

I contacted the problem author, regarding the intended score: 
```
According to task 5, the best solution was obtained with additional model fine-tuning on the validation dataset after initializing zero embeddings in a manner similar to the approach described in the repository. Therefore, we did not take this solution into account in our ranking. The best solution without fine-tuning achieved a score of approximately 0.44.  
```

### Task 6: [The Hogspell Challenge](https://www.kaggle.com/code/lenjjiv/en-hogspell-baseline-solution)

Summary: fine-tune stable diffusion 1.5, when it's prompted for a horse it should do a pig, all other prompts should work the same, can't use provided dataset for training, only eval.

Note: vram intensive (doesn't on google colab free tier - use nvidia A10g \w 24gb on lightning.ai, 10 free hours)

### Task 7: [Cuties Segmentation](https://www.kaggle.com/code/tatianagaintseva/baseline-eng)

Summary: given a pretrained CLIP, produce binary segmentation masks for cats & dogs of different breeds, training on validation data is allowed

### Task 8: [Intent Detection and Slot Filling](https://www.kaggle.com/code/ilseyaralimova/baseline-for-nlp-task)

Summary: train a joint BERT model for intent classification and slot filling, there are train/validation/test datasets.
