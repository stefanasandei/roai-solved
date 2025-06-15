# IAIO 2024

## Task 1: metrics

1 a. Accuracy does not take into consideration imbalanced classes, treating them equally, which is highly problematic when the target class ("has disease") is in the minority.

A better metric is recall (true positive rate), which measures how many instances of the target class were found (it's crucial we identify as many people with the disease as possible, better to have false posivitives than false negatives). Example: 95% recall means only 5% of diseased patients are missed.

1 b. A model that only predicts "false" will get a 95% accuracy here, a pretty high score.

2 a. F1-Score is best here

2 b. F1-score is the harmonic mean of precision (avoiding false positives) and recall (catching true positives). This is critical here because:

- Precision minimizes misclassifying legitimate emails as spam (false positives), ensuring important emails aren’t lost.

- Recall maximizes detecting actual spam (true positives), reducing inbox clutter.

2 c. Precision will minimize the false positives, however this can lead to labeling all data as non-spam (1 precision, but 0 recall). Recall will try and find all instances of spam, leading to good emails being labeled as spam. A mix of the two leads the best results

Resources to learn:

- https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall

## Task 2: ethics, probability

#### 1. Trolley Problem with Probabilities

**Good answers: C, D**

- **C**: Neither option saves more lives.
  - _Reason_: Expected lives lost:
    - Don’t pull: $0.5 \times 2 = 1.0$
    - Pull: $0.1 \times 10 = 1.0$  
      Both choices have identical expected outcomes.
- **D**: It is a moral dilemma. It shouldn’t be solved with a probability equation.
  - _Reason_: The trolley problem involves ethical trade-offs (e.g., risk distribution, uncertainty about outcomes) beyond pure probability.

#### 2. Biased Job Applicant

**Good answers: B, C, D**

- **B**: Adjust the AI to remove bias, risking historical consistency.
  - _Reason_: Actively correcting bias is ethical, even if it disrupts flawed historical patterns.
- **C**: Collect new diverse data and retrain, potentially delaying hiring.
  - _Reason_: Long-term fairness justifies short-term delays.
- **D**: Combine adjustments/retraining while weighing costs.
  - _Reason_: A balanced approach addresses bias pragmatically.
- _Excluded A_: Ignoring bias perpetuates discrimination.

#### 3. Ethical Self-Driving Car

**Good answers: C, D**

- **C**: Follow programmed ethical guidelines prioritizing overall harm minimization.
  - _Reason_: Ethical frameworks should be predefined and harm-focused (e.g., utilitarianism).
- **D**: Make a random decision to avoid bias.
  - _Reason_: Avoids discriminatory outcomes (e.g., age-based discrimination).
- _Excluded A/B_: Choosing based on age (remaining lifespan/recovery) is unethical and potentially illegal.

#### 4. Hospital’s Gender-Biased AI

**Good answers: A, B, C**

- **A**: Adjust algorithms to improve accuracy for women, risking side effects.
  - _Reason_: Proactive correction is necessary despite potential trade-offs.
- **B**: Retrain with balanced data, accepting delays.
  - _Reason_: Ensures equitable care across genders.
- **C**: Combine retraining/adjustments, accepting higher costs.
  - _Reason_: A multifaceted approach maximizes fairness.
- _Excluded D_: Continuing use compromises women’s health (unethical).

#### 5. Surveillance Dilemma

**Good answers: A, B, D**

- **A**: Modify AI to eliminate bias, accepting reduced effectiveness.
  - _Reason_: Prevents harm from discriminatory targeting.
- **B**: Implement oversight, requiring resources.
  - _Reason_: Human monitoring ensures accountability.
- **D**: Combine modifications/oversight, balancing fairness/resources.
  - _Reason_: Practical and ethical middle ground.
- _Excluded C_: Keeping the system perpetuates injustice.

#### 6. Transparent Social Media Algorithm

**Good answers: B, C, D**

- **B**: Disclose fully to ensure transparency and trust.
  - _Reason_: Builds user trust and accountability.
- **C**: Partially disclose to balance transparency/proprietary interests.
  - _Reason_: Mitigates risks while addressing concerns.
- **D**: Conduct independent audits instead of full disclosure.
  - _Reason_: Verifies fairness without exposing trade secrets.
- _Excluded A_: Non-disclosure erodes public trust.

#### 7. Ethical Use of Copyrighted Material in AI

**Good answers: A, B, C**

- **A**: Use only explicitly free-to-use data.
  - _Reason_: Avoids legal/ethical violations.
- **B**: Negotiate agreements with content owners.
  - _Reason_: Ensures fair compensation/permission.
- **C**: Track and credit original creators.
  - _Reason_: Recognizes contributions and aligns with attribution ethics.
- _Excluded D_: Non-commercial data sources are often insufficient and impractical.

## Task 3: kmeans

1\. Compute the distance from each point to the two centroids. Assign the cluster based on the minimum distance.

2\. Compute the mean of the point coordonates from each clusters A and B. The resulted mean x, y will be the coordonates for the new centroids.

3\. Define the **Sum of Squared Errors (SSE)** for partition $P$ as:

$$\text{SSE}(P) = \sum_{i=1}^k \sum_{x \in C_i} \|x - \mu_i\|^2$$

where $C_i$ is cluster $i$ and $\mu_i$ is its centroid. Sum of all distances from all $k$ clusters of all points.

First we study the **monotony**. In the assignment step, points are reassigned to the nearest centroid. For fixed centroids, this minimizes SSE. Thus:

$$\text{SSE}(P_{t+1}^{\text{assign}}) \leq \text{SSE}(P_t)$$

In the update step, centroids are recomputed as cluster means. For fixed clusters, this minimizes SSE. Thus:

$$\text{SSE}(P_{t+1}) \leq \text{SSE}(P_{t+1}^{\text{assign}})$$

Consequently:

$$\text{SSE}(P_{t+1}) \leq \text{SSE}(P_t) \text{ (1)}$$

with strict decrease if $P_{t+1} \neq P_t$

Suppose the algorithm revisits a configuration $P$ at iterations $t$ and $s$ ($s > t$). Then:  
$$\text{SSE}(P_t) = \text{SSE}(P_s)$$
However, if $P_{t+1} \neq P_t$, strict decrease implies: $$\text{SSE}(P_t) > \text{SSE}(P_{t+1}) > \cdots > \text{SSE}(P_s) = \text{SSE}(P_t)$$

a **contradiction**. Thus, configurations cannot repeat unless $P_{t+1} = P_t$ (convergence).

**Finite Convergence:**

- The sequence $\{\text{SSE}(P_t)\}_{t=0}^\infty$ is **non-increasing** and bounded below (by 0).
- Since $\mathcal{P}$ is finite and no configuration repeats, the algorithm must **terminate** when:  
  $$P_{t+1} = P_t$$
  (i.e., no points are reassigned).
- The maximum number of steps is bounded by $|\mathcal{P}| \leq k^n$. (number of valid partitions of n points into k clusters)

## Task 4: neural networks

1\. What is the parameter count of a linear layer that takes as input a one-hot $V=8192$ vector and outputs an embedding of size $d=512$?

The parameter count of the weights is $w = V \cdot d$ and there is no bias. Example: (1, 8192) @ (8192, 512) = (1, 512)

2\. First linear layer has input dimension $d$ and output dimension $f = 2048$. Second layer has input $f$ and output $d$.

Parameter count for the weights of the first layer is $w_1 = d \cdot V$, for the bias $b_1 = V$, so total parameter count is $(d+1) \cdot V$.

Parameter count for the weights of the first layer is $w_2 = V \cdot d$, for the bias $b_2 = d$, so total parameter count is $(V+1) \cdot d$.

## Task 5: perceptron

-

## Task 6: reinforcement learning

-
