# ONIA Winter Warmup Challenge 2025

## The Glitch Hunter: 98.57p

TLDR: given images with "glitches" on top of them (rectangles either with noise, inverted colors, etc.), compute a binary mask showing where the glitches are.

Solution:
- train a UNet to predict the binary masks
- use data augmentation, I used separate transforms for color augmentation and geometric augmentations (those that do affect the mask and need to be "in sync")
- add dropout and batch normalization to the model, for a more stable long training run
- depending on your hyperparameters, the training can take from 5 to 100 epochs. For a decent score of 80-90 IoU you don't need much training, but I experimented with long training runs and learning rate decay towards the end to maximize the score. Before running my notebook, experiment yourself with the hyperparameters (no AdamW weight decay, lr of 1e-3 and 10 epochs is a good starting point)
- for faster training & inference, work with images resized to 128x128 or 256x256. Issue is when upscaling for the submission you will get blurrier results, for this I used the fact that all glitches are rectangles, I wrote a function to find all components and draw them as perfect squares (experiment with the min_area for better results).
- also for tiny better results, I used adaptive threshold: iterate over threshold, evaluate on the validation dataset and choose the best one

I think that a perfect score might be achievable using test time augmentations, model soup, predictions post-processing and adaptive threshold.

## The Magic of Words: 100p

TLDR: Given a 1024×1024 image and a list of 20 candidate words (3 of which were actually used to generate the image, drawn from 3 different thematic categories), the task is to return the five most relevant words in ranked order.

Solution 70p (`magic-pretrained.ipynb`):
- use a pretrained CLIP (`openai/clip-vit-base-patch16`)
- feed it the image and its words, no need for templates
- sort by their probabilities and get the answer

This works very nicely, since CLIP was trained to assign similarities between an image and its possible captions. The higher the cosine similarity between a text and the image, the higher the likelihood its a correct caption, so we also get the ranking order.

Solution 100p (`magic-ft.ipynb`):
- also use CLIP, but this time we finetune it
- used binary cross entropy loss

CLIP works very well out of the box, so we just have to nudge it a bit in our direction. For the labels, treat them as un-normalized scores, not cosine similarities, assign 1 to correct words (maximum probability) and 0 to wrong words. So now CLIP simply learns to output high logits for positives (label 1) and low logits for negatives (label 0).

## The National Rabbit Exhibition: 100p

TLDR: We are given a dataset of tagged rabbits and 3 subtasks:  
1. Count how many females have lopped ears and Havana-colored fur.  
2. Recover the 3 missing breed labels by clustering the rabbit.
3. Predict the missing judging scores (0–100) for unscored rabbits.

Solution subtask 1:
- basic pandas filtering, create 3 masks and `&` them
  
Solution subtask 3:
- cluster based on all available features
- use kmeans with 3 clusters

Solution subtask 3:
- scale the data
- apply one hot encoding
- random forest solves the deal

