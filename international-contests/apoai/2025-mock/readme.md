# APOAI2025 Mock Competition

Link: https://www.bohrium.com/competitions/7647426696

## Status

| Task | Score      | Place % | Type |
| ---- | ---------- | ------- | ---- |
| 1    | **0.6171** | **82%** | ML   |
| 2    | **0.9959** | **81%** | CV   |
| 3    | **0.9597** | **92%** | NLP  |
| 4    | **0.9069** | **92%** | ML   |

## Explanations

### Task 1: [Predicting the Shooting Percentage of Basketball Stars](https://www.bohrium.com/competitions/5135119121?tab=introduce)

Summary: Given a dataset of x and y location coordonates, we have to predict if a basketbasll shot was made or not. We have to use an MLP

Solution: train a 3 layer MLP, with dropout, for 5 epochs

### Task 2: [Real or Fake Image Recognition Task](https://www.bohrium.com/competitions/2623226705?tab=introduce)

Summary: Given a dataset of real (from Cifar10) and fake (generated using AI) images, we have to train a CNN classifier

Solution: create a model with 2 conv layers (3 -> 8 -> 16 channels) and 2 fully connected layers, train for 10 epochs; a learning rate scheduler was used for more stable training

### Task 3: [News Text Classification Task](https://www.bohrium.com/competitions/2223242868?tab=introduce)

Summary: Given a text dataset of news, classify items into categories

Solution: tfidf + logistic regression

### Task 4: [Solving the Pendulum Motion with Missing Data](https://www.bohrium.com/competitions/1723157880?tab=introduce)

Summary: A pendulum is executing a motion and we measure the angle theta at time t (the dataset). At an unknown time an external force is started being applied upon the ball, at the end of the string. The measuring device also stopped measuring for a random time segment. We have to approximate the parameters of the differential equation of the motion.

Solution: Use a physics-informed neural network, that uses the formula given in the task statement, and use regression on this to find $\alpha$, $\beta_1$, $\beta_2$. The math is explained in the task statement, a summary of the variables is also given in the notebook.
