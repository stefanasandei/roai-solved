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

