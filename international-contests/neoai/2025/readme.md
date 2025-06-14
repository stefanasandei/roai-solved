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
| 8    | **0.34** | 0.29     | 0.58       | **17%**    | NLP  |

## Explanations

### Task 1: [Tracy tables](https://www.kaggle.com/code/timriggins/basel1ne-tricy-table-data)

Summary: given a table with plenty unknown features, do feature engineering to increase the score of a lightgbm model

Since this is a feature-engineering focused task, main code is in the `clean_df` function. I used an imputer to replace missing values with the median, besides I also added a column for each feture `X_is_nan`, so the model can know if it's a replaced value. I dropped some columns, which had low correlation (see `.corrwith`). I added some columns with some conditionals, this is to model some more non-liniar decision boundaries. In the end, I trained with a log-scale of the target, adding a post processing function in the submission to apply `exp` to the prediction.


### Task 2: [Underfitting CV](https://www.kaggle.com/code/timriggins/baseline-cv-underfitting)

Summary: given a vision transformer trained on 90 classes and images corresponding to 10 other classes, fine-tune the ViT on the new classes, without degrading performance too much on the 90 classes. There is no access to images from 90 classes. 

Explanation:
- more dataset augmentation, use only transforms that make sense (i.e. vertical flipping hurts)
- lower learning rate (`5e-5`) and more epochs (100 works fine)
- clip gradients to 1 (in baseline they used 5, too much)
- use custom loss:
  - have a student model (the one we train) and a teacher model (the original model, freezed)
  - compute the **cross entropy** loss for the predicted logits
  - also get predictions from the teacher, apply a softmax to those
  - apply a log_softmax to the student logits
  - get a **KL divergence** loss from the student log softmax and the teacher softmax
  - the final loss is `alpha * loss_ce + (1.0 - alpha) * loss_kd`, I used 0.7 for alpha, but you can try to tweak it

### Task 3: [Evading AI-Generated Text Detection](https://www.kaggle.com/code/egorgij21/baseline)

Summary: given a gemma2 2b model, change its outputs to be more human-like, according to a given dataset, do not finetune

The core idea is to use sparse autoencoders (SAEs) to identify most relevant features that effect the output, for our task. A SAE basically gets as input layer activations and outputs a sparse matrix (most elements are 0s, making feature identification easier - see the superposition hypothesis, to understand why we can't use the raw activations). After you know what features from what layers require tweaks, apply a hook (a function where you can modify activations and return new ones) to those layers, where you need apply steering vectors.

Resources to learn more:
- "Golden Gate Claude", blog article: https://www.anthropic.com/news/golden-gate-claude
- "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet", paper: https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
- youtube video explaining the above: https://youtu.be/QqrGt5GrGfw ("I Am The Golden Gate Bridge & Why That's Important." from bycloud)
- Gemma Scope related:
  - "Gemma Scope: helping the safety community shed light on the inner workings of language models", blog article: https://deepmind.google/discover/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/
  - coding tutorial: https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp?usp=sharing
  - interactive demo: https://www.neuronpedia.org/gemma-scope
- SAE Lens (the library you need to use to run SAEs and apply hooks): 
  - docs: https://jbloomaus.github.io/SAELens/latest/
  - coding tutorials: https://github.com/jbloomAus/SAELens/tree/main/tutorials

As of yet, I have now written my own solution.

Note: vram intensive (doesn't works on google colab free tier); used gpt2-small instead of gemma2 2b

### Task 4: [Cluster images](https://www.kaggle.com/code/timriggins/basel1ne-cluster-1mages)

Summary: given arrays of heavily augmented images, which come from original 32 images, cluster the images into 32 clusters

This is not really deep-learning related, it's based on a trick. We have two arrays, X1 of shape (N, 128, 4) and X2 of shape (N, 4, 128). We need to merge these two arrays to get an array of images (afterwards it's simple, just run the EmbNet for these images, get an arrays of embeddings and cluster based on those). We use `torch.bmm` (batch matrix multiplication - https://docs.pytorch.org/docs/stable/generated/torch.bmm.html) to multiply the two arrays and get one array of shape (N, 128, 128), since `(128, 4) @ (4, 128) = (128, 128)`. Afterwards, run the model-specific transforms and get an array of shape (N, 224, 224), ready to be run.

### Task 5: [Broken BERT](https://www.kaggle.com/code/ilseyaralimova/broken-bert-baseline)

Summary: given a BERT model with broken embeddings (some tokens have embeddings fully null), fix the embeddings without fine-tune.

I contacted the problem author, regarding the intended score: 
```
According to task 5, the best solution was obtained with additional model fine-tuning on the validation dataset after initializing zero embeddings in a manner similar to the approach described in the repository. Therefore, we did not take this solution into account in our ranking. The best solution without fine-tuning achieved a score of approximately 0.44.  
```

Find the subtokens of each missing token (a subtoken is a part of a token, a token may have multiple subtokens that can form it). Do the mean of those subtokens and assign it to each missing token. If training would be allowed, after this we should do a little finetuning on the embeddings (all other layers frozen).

### Task 6: [The Hogspell Challenge](https://www.kaggle.com/code/lenjjiv/en-hogspell-baseline-solution)

Summary: fine-tune stable diffusion 1.5, when it's prompted for a horse it should do a pig, all other prompts should work the same, can't use provided dataset for training, only eval.

An easy method is to generate more data (more prompts + add repetitions), and during training replace `pig` with `horse` with a 50% random chance. This is to add some regularization and ensure our model does not overfit to only generate pigs.

Another main idea is from the DreamBooth paper (https://arxiv.org/abs/2208.12242), mainly the loss. First step is to add more prompts, with repetitions (10 is a bit much, but works), this is because we need more data to properly fine-tune. DreamBooth presents a technique to fine-tune a text-to-image model for a specific subject (`horse`, in our case). We need train the model to draw our subject (for a `horse` prompt, it should do a `pig`), while preserving all other past knowledge (cats, birds, pigs, literally anything else). 

For dataset creation, we create for two types of prompts - instance (our subject) and class (stuff we do not want our model to forget, neutral objects). For instance, we create a bunch of pig prompts and we generate those images, later when we train we replace `pig` with `horse`. For class, we need a matching number of images with neutral content. 

During training, we run the instance prompt and get our noise. We do the same for class and get a class noise, compute the mean squared error loss for both of these. Final loss is `instance_loss + lambda * class_loss`, where lambda can be 1.0, or 1.5 for stronger regularization.

Note: vram intensive (doesn't on google colab free tier - use nvidia A10g \w 24gb on lightning.ai, 10 free hours)

### Task 7: [Cuties Segmentation](https://www.kaggle.com/code/tatianagaintseva/baseline-eng)

Summary: given a pretrained CLIP, produce binary segmentation masks for cats & dogs of different breeds, training on validation data is allowed

CLIP is trained with contrastive learning to predict the similarity between images and texts. We only use the vision encoder of CLIP, it's useful since it was trained to have bigger activations for the subject (aka the region of the image that will lead to a correct prediction), in the `get_clip_activations` we return the activations from several layers (since it's layer has a different task, we want to gather as much relevant features as possible) for a given image. The image is split into 16 by 16 patches, so we will have (224/16)^2=196 patches, each with 768 features. I used in the solution 4 layers from the vision encoder.

We then train a classifier model on 768*4 features to predict wheter the subject is present there (1 in the binary mask, we get the binary masks from the validation dataset and we resize them to 14x14 - each pixel from the resized masks corresponds to a target class). When we generate the mask, we run the get the predicted probabilities for each of the 14x14 pixels, and we put those in a heatmap. Interpolate the heatmap to (224, 244), apply a threshold to convert it to a binary mask and run 1 iteration of binary dilation. 

The best threshold is found by running several options over the validation dataset and computing the intersaction over union (IoU) score.

In the baseline, they also computed text embeddings, using the text encoder. Then they found the target class for each image (using the dot product and choosing the greatest value), and they stored the embedding value for the class. Conver the image into patches and get the similarity scores between the image and the target class. The probabilites are the heatmap (as in imy solution), the rest is like in the above.

### Task 8: [Intent Detection and Slot Filling](https://www.kaggle.com/code/ilseyaralimova/baseline-for-nlp-task)

Summary: train a joint BERT model for intent classification and slot filling, there are train/validation/test datasets.

The issue is due the fact that the training dataset is in english and the testing dataset is in russian. I did not finish coding this task, however it is worth trying to seach in `ru_en_pairs.jsonl` for translations and replace where possible. This way we can train in russian. Another thing recommended by chatgpt was to do "Unsupervised Domain Adaptation via Continued MLM Pretraining", basically use `unlabeled_texts.txt` to train the BERT's text encoder on the russian language, use the `DataCollatorForLanguageModeling` collator.

Another basic ideas would be to tweak the epochs number, learning rate, gradient clipping and add a learning rate scheduler. We can't change the loss, so no modifications there. 