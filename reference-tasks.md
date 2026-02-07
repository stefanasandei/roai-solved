# Reference Tasks

An outline of the most useful tasks to learn competitive AI. Take it as a roadmap to learn topics.

Each table goes from easiest tasks to "most interesting". Do the tasks in order, they will take you through 90% of the required knowledge. I didn't duplicate topics, for example you *might* need to do more tasks with topic X to understand it, go to [solutions](./reference-solutions.md) to see the gist of all tasks and find what you want to practice.

## Tabular Machine Learning

Quick roadmap to get you familar with classical machine learning models and tabular dataset processing. You'll start by doing dataset exploration (see [EDA](https://en.wikipedia.org/wiki/Exploratory_data_analysis)), progress to feature engineering (when your model can't figure it out on its own so you ground it with useful features), continue with modelling (linear regression, support vector machines, random forest, gradient boosting and MLPs), then deeper EDA (removal of outliers, data normalization, etc.) and end with some *brief* theory tasks.

| competition                 | task                    | what you'll learn                        |
| --------------------------- | ----------------------- | ---------------------------------------- |
| OJIA 2025                   | smart cargo             | hello world - basic linear regression    |
| BCS Easter Round 2025       | cybersecurity ai        | data cleaning                            |
| Local Stage Simulation 2026 | transport               | clustering (1)                           |
| IOAI Day 2 2025             | antique                 | clustering (2) + data exploration        |
| Simulare OJIA 1 2025        | credit score            | feature engineering (1)                  |
| Simulare OJIA 2 2025        | status pacient          | feature engineering (2)                  |
| Local Stage Simulation 2026 | paintings               | data cleaning + feature engineering (3)  |
| "Unirea" College Round 2026 | bac under scrutiny      | FE for time series                       |
| Simulare OJIA 2 2025        | pret casa               | models (1) + FE recap                    |
| Local Stage Simulation 2026 | parkinson               | models (2)                               |
| BCS Easter Round 2025       | brain anomaly detection | models (3)                               |
| ONIA Winter Warmup 2025     | rabbit exhibition       | models (4) + data cleaning               |
| IOAI Mock 2025              | chem simulation         | normalization, outliers, FE + models (5) |
| Georgia Round 1 2026        | memory trace            | model theory (1)                         |
| Georgia Round 1 2026        | planet x                | model theory (2)                         |

Resources (easy, short -> hard, long), some might be overkill in the beginning, but if you read this you're in for the full ride:
- Google's [ML crash course](https://developers.google.com/machine-learning/crash-course)
- ["A Recipe for Training Neural Networks"](https://karpathy.github.io/2019/04/25/recipe/) by Andrej Karpathy: ***Become one with the data***
- "The Hundred-Page ML Book"
- Scikit Learn's [User Guide](https://scikit-learn.org/stable/user_guide.html) - read this like a book
- Books: [Mathematics for ML](https://mml-book.github.io/book/mml-book_printed.pdf), [Probabilistic ML: An Introduction](https://github.com/probml/pml-book/releases/latest/download/book1.pdf), [CS229 Lecture Notes](https://cs229.stanford.edu/notes2022fall/main_notes.pdf), ["Exerciţii de învăţare automată"](https://piazza.com/class_profile/get_resource/mfy56qgohaq65z/mfy63lst4lj3k3)

If you want more practice, I recommend the ML tasks from [AICC](https://ioai-community-contest.netlify.app/contests) (their focus is on EDA and theory), as well as the OJIA-level ML rounds from [Nitro Judge](https://judge.nitro-ai.org/competitions).

## Natural Language Processing

Starting from beginner with texts in tabular datasets, to working with deep learning generative models. For language processing, you can use NLTK.

| competition             | task                                                               | what you'll learn                           |
| ----------------------- | ------------------------------------------------------------------ | ------------------------------------------- |
| Nitro NLP 2025          | mickey si donald                                                   | tfidf, models (1)                           |
| BCS Intermediate 1 2025 | are you a robot                                                    | language processing, tfidf                  |
| ONIA 2025               | om vs ai                                                           | text cleaning (1) + tfidf, models (2)       |
| ROAI 2025               | toxic                                                              | text cleaning (2) + tfidf, models (3)       |
| MLCompete               | verses author                                                      | nlp feature engineering, text cleaning (3)  |
| Algolymp Winter Round   | idk man                                                            | named entity recognition                    |
| ROAI 2025               | skeletons                                                          | recurrent neural networks, sequence data    |
| Poland Phase 2 2025     | source extraction                                                  | embeddings (1)                              |
| MLCompete               | [Residency Exam](https://platform.olimpiada-ai.ro/en/problems/43)  | embeddings (2) - but with mid score         |
| NEOAI 2025              | broken bert                                                        | embeddings (3)                              |
| ROAI 2025               | toxic (again)                                                      | model finetuning (1), Bert                  |
| MLCompete               | [Text correction](https://platform.olimpiada-ai.ro/en/problems/46) | model finetuning (2), T5                    |
| AICC Round 2            | essay gap                                                          | model finetuning (3), Bert                  |
| MLCompete               | Residency Exam (again for full score)                              | LLMs - local vLLM inference or using an API |
| NEOAI 2025              | evading ai text detection                                          | LLM steering + sparse autoencoders          |

## Computer Vision

For image processing tasks, I highly recommend solving them using *only* OpenCV. 

| competition                 | task                          | what you'll learn                  |
| --------------------------- | ----------------------------- | ---------------------------------- |
| MLCompete                   | Real Art vs. AI-Generated Art | CNN classification                 |
| MLCompete                   | Saving Christmas              | pretrained CNN regression          |
| ROAI 2025                   | hotspot                       | image preprocessing (1)            |
| ONIA 2025                   | notatie bizantina             | image processing (2)               |
| Decebal Tech 2025           | sami                          | image processing (3)               |
| Algolymp PreOJIA 11-12 2026 | lunar craters                 | image processing (4)               |
| Poland Phase 2 2025         | non-normal dist               | encoder-decoder architecture       |
| ONIA Winter Warmup 2025     | glitch hunter                 | binary segmentation using UNet (1) |
| IOAI At-Home 2025           | weather                       | UNet (2) + advanced modules + TTA  |
| IOAI At-Home 2025           | radar                         | encoder-decoder, focal loss        |
| Spooky Round 2025           | haunt me                      | object detection (1), faster rcnn  |
| Nitro NLP 2025              | find the ducks                | object detection (2), yolo         |
| AICC Round 2                | face matching                 | CLIP (1), embeddings retrieval     |
| ONIA Winter Warmup 2025     | magic of words                | CLIP (2), finetune                 |
| NEOAI 2025                  | cuties segmentation           | CLIP (3), CLIPSeg                  |
| IOAI Day 2 2025             | restroom                      | CLIP (4), CLIPReID                 |
| IOAI Day 2 2025             | pixel                         | CLIP (5), MaskCLIP                 |
| NEOAI 2025                  | hogspell challenge            | stable diffusion finetune          |
| MLCompete                   | thousand rooms                | vqa                                |

## Audio

| competition       | task           | what you'll learn             |
| ----------------- | -------------- | ----------------------------- |
| AICC Round 3      | soud of nature | melspectogram classification  |
| Spooky Round 2025 | creepy pizza   | lstm for sound classification |
| AICC Round 2      | demixing audio | unet for sounds               |

## Deep Learning

| competition      | task              | what you'll learn                         |
| ---------------- | ----------------- | ----------------------------------------- |
| NEOAI 2025       | underfitting cv   | hyperparam tuning + KV divergence loss    |
| AICC Round 0     | find brain tumors | self supervised learning (BYOL, FixMatch) |
| IOAI Mock 2025   | grid collage      | pseudo labeling                           |
| AICC Round 1     | the defected nuts | anomaly detection                         |
| CEOAI Practice 1 | star observatory  | knowledge distillation                    |