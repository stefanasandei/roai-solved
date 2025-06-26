# Polish AI Olympiad 2025 - Phase 2

https://github.com/OlimpiadaAI/II-OlimpiadaAI/tree/main/2_etap

In the second stage, the participants competed by solving the following tasks:

- Source Extraction: selection of useful embedding in searching for source texts.
- Borrowing: finding directions in the data space that most effectively change classifier decisions.
- Non-normal Distribution: removal and classification of various types of noise in the image.

## Status

| Task | Score    | Type |
| ---- | -------- | ---- |
| 1    | **88%**  | NLP  |
| 2    | **-**    | ML   |
| 3    | **100%** | CV   |

## Explanations

### Task 1: [Source Extraction](https://github.com/OlimpiadaAI/II-OlimpiadaAI/blob/main/2_etap/ekstrakcja_zrodel/ekstrakcja_zrodel.ipynb)

Summary: a RAG system requires an Embedder class that exposes two methods: `encode_queries` (a list of text) and `encode_corpus` (a list of documents with title and content). We have to implement the Embedder class, using only a pretrained [SGPT](https://arxiv.org/abs/2202.08904) model.

Explanation coming soon!

### Task 2: [Borrowing](https://github.com/OlimpiadaAI/II-OlimpiadaAI/blob/main/2_etap/kredytobranie/kredytobranie.ipynb)

Summary: -

Explanation coming soon!

### Task 3: [Non-normal Distribution](https://github.com/OlimpiadaAI/II-OlimpiadaAI/blob/main/2_etap/rozklad_nienormalny/rozklad_nienormalny.ipynb)

Summary: a syntethics grayscale image dataset has been created, later noise (uniform or gaussian) has been added to it. Given an image, we have to implement a model that predicts three things: a denoised image, type of the noise, and if it's gaussian it's parameters (mu and sigma).

Explanation coming soon!
