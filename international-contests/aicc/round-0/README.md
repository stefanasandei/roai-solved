# AICC Round 0

| metric      | 01 Deceptive Points | 02 Brain Tumor | 03 Latent Model Classification |
| ----------- | ------------------- | -------------- | ------------------------------ |
| score       | 0.08244             | 0.78010        | 0.98853                        |
| leaderboard | -                   | -              | 0                              |

Explanations coming soon, check out Nikoloz Gegenava's write ups in the meantime: https://ioai-community-contest.netlify.app/community

However you should also be able to understand the solutions by reading the notebooks.

### 01. Deceptive Points

Summary: pca, kmeans + huber regression

Explanation: todo

### 02. Brain Tumor

Summary: do a pretraining with an SSL framework to learn the embeddings (BYOL was used in the solution), then do a finetune on the few labeled to get the last fully connected layers to learn the embeddings to class correlation; another finetune using pseudo labels can also boost the F1 score by a few points.

SSL frameworks that work: BYOL, FixMatch, VICReg, MuCo. SimCLR isn't recommended since it requires a huge batch size.

Explanation: todo

### 03. Latent Model Classification

Summary: todo

Explanation: todo
