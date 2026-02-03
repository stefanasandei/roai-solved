# Reference Solutions

This file contains the short idea to solve all tasks I did.

## National Olympiads

| country | round           | task                       | core idea                                          | type  |
| ------- | --------------- | -------------------------- | -------------------------------------------------- | ----- |
| Romania | Regional 2025   | consum caloric             | feature engineering (poly), linear regression      | ML    |
| Romania | Regional 2025   | smart cargo                | feature engineering (dummy), linear regression     | ML    |
| Romania | National 2025   | notatie bizantina          | image processing, CNN classifier                   | CV    |
| Romania | National 2025   | om vs ai                   | text cleaning, tfidf, clustering                   | NLP   |
| Romania | Camp CV 2025    | angry birds                | work in progress                                   | CV    |
| Romania | Camp CV 2025    | hotspot                    | image preprocessing                                | CV    |
| Romania | Camp NLP 2025   | skeletons                  | LSTM model, time series processing                 | NLP   |
| Romania | Camp NLP 2025   | toxic                      | text cleaning, tfidf, catboost, threshold tuning   | NLP   |
| Poland  | Phase 1 2025    | coin counting machine      | work in progress                                   | CV    |
| Poland  | Phase 1 2025    | hallucination detection    | work in progress                                   | NLP   |
| Poland  | Phase 1 2025    | ECG signal disturbances    | work in progress                                   | ML    |
| Poland  | Phase 1 2025    | noise in data labels       | work in progress                                   | CV    |
| Poland  | Phase 1 2025    | hidden substrings          | work in progress                                   | NLP   |
| Poland  | Phase 2 2025    | borrowing                  | work in progress                                   | ML    |
| Poland  | Phase 2 2025    | non-normal dist            | multi-task CV model + encoder-decoder architecture | CV    |
| Poland  | Phase 2 2025    | source extraction          | retrieval using embeddings                         | NLP   |
| Romania | Local ROAI 2026 | calitate sol               | dataframe operations & cleaning, FE based on that  | ML    |
| Romania | Local ROAI 2026 | anpc                       | bfs on state space                                 | ML    |
| Georgia | ML Round 1 2026 | data reduction             | work in progress                                   | ML    |
| Georgia | ML Round 1 2026 | memory trace               | model confidence threshold                         | ML    |
| Georgia | ML Round 1 2026 | planet x                   | knn on lr coefs, find k most similar models        | ML    |
| Poland  | Phase 1 2026    | convolution filters        | work in progress                                   | CV    |
| Poland  | Phase 1 2026    | multi-label classification | work in progress                                   | CV    |
| Poland  | Phase 1 2026    | whisper or scream          | work in progress                                   | Audio |
| Poland  | Phase 1 2026    | semantic changes           | work in progress                                   | NLP   |
| Poland  | Phase 1 2026    | multispectral segmentation | work in progress                                   | CV    |

## International Contests

| contest           | task                              | core idea                                       | type |
| ----------------- | --------------------------------- | ----------------------------------------------- | ---- |
| NEOAI 2025        | tracy tables                      | feature engineering                             | ML   |
| NEOAI 2025        | underfitting cv                   | hyperparam tuning + KV divergence loss          | CV   |
| NEOAI 2025        | evading ai text detection         | LLM steering + sparse autoencoders              | NLP  |
| NEOAI 2025        | cluster images                    | torch.bmm trick                                 | ML   |
| NEOAI 2025        | broken bert                       | fix null embeddings with sub-word embeddings    | NLP  |
| NEOAI 2025        | hogspell challenge                | stable diffusion finetune (dreambooth training) | CV   |
| NEOAI 2025        | cuties segmentation               | classifier for CLIP activations (CLIPSeg)       | CV   |
| NEOAI 2025        | intent detection and slot filling | work in progress                                | NLP  |
| APOAI Mock 2025   | basketball                        | MLP training                                    | ML   |
| APOAI Mock 2025   | text                              | tfidf                                           | NLP  |
| APOAI Mock 2025   | pendulum                          | physics informed neural network                 | ML   |
| APOAI Mock 2025   | classifier                        | CNN training                                    | CV   |
| IOAI At-Home 2025 | radar                             | encoder-decoder CNN, focal loss                 | CV   |
| IOAI At-Home 2025 | weather                           | UNet, ASPP, FiLM, dice loss, dynamic threshold  | CV   |
| IOAI At-Home 2025 | chameleon                         | prompt engineering                              | NLP  |
| IOAI Mock 2025    | dark matter                       | obj detection, tta                              | CV   |
| IOAI Mock 2025    | chem simulation                   | FE (interactions), huber, norm, lr schedule     | ML   |
| IOAI Mock 2025    | grid collage                      | pseudo labeling                                 | CV   |
| IOAI Day 1 2025   | radar                             | work in progress                                | CV   |
| IOAI Day 1 2025   | chicken counting                  | work in progress                                | CV   |
| IOAI Day 1 2025   | concepts                          | work in progress                                | NLP  |
| IOAI Day 2 2025   | restroom                          | work in progress                                | CV   |
| IOAI Day 2 2025   | antique                           | work in progress                                | ML   |
| IOAI Day 2 2025   | pixel                             | work in progress                                | CV   |

[AI Community Contest](https://ioai-community-contest.netlify.app/contests):

| contest      | task                        | core idea                    | type  |
| ------------ | --------------------------- | ---------------------------- | ----- |
| AICC Round 0 | deceptive points            | pca + eda                    | ML    |
| AICC Round 0 | find brain tumors           | SSL algos: BYOL, FixMatch    | CV    |
| AICC Round 0 | latent model classification | ridge, ml theory             | ML    |
| AICC Round 1 | is that audio               | basic cls                    | Audio |
| AICC Round 1 | autocorrect                 | work in progress             | NLP   |
| AICC Round 1 | the defected nuts           | anomaly detection            | CV    |
| AICC Round 2 | face matching               | clip embeddings retrieval    | CV    |
| AICC Round 2 | demixing audio              | unet to predict 2 sounds     | Audio |
| AICC Round 2 | essay gap                   | bert finetune                | NLP   |
| AICC Round 3 | morty time paradox          | work in progress             | ML    |
| AICC Round 3 | soud of nature              | melspectogram classification | Audio |
| AICC Round 3 | drawn apart                 | work in progress             | CV    |

Note: these are my personal solutions, for official solutions go [here](https://github.com/AI-Community-Contest/solutions).

## Nitro Judge Contests

| competition               | task                    | core idea                                                           | type   |
| ------------------------- | ----------------------- | ------------------------------------------------------------------- | ------ |
| Simulare OJIA 1 2025      | livrare pachete         | plain linear regression                                             | ML     |
| Simulare OJIA 1 2025      | credit score            | FE (dummy encoding, drop useless, date parsing)                     | ML     |
| Nitro NLP 2025            | mickey si donald        | tfidf, linear svc + nb                                              | NLP    |
| Nitro NLP 2025            | find the ducks          | finetune yolo                                                       | CV     |
| BCS Easter Round 2025     | cybersecurity ai        | datetime features, fill missing                                     | ML     |
| BCS Easter Round 2025     | brain anomaly detection | basic image classifier                                              | CV     |
| Simulare OJIA 2 2025      | status pacient          | FE (drop useless, ordinal encoding, dummy encoding)                 | ML     |
| Simulare OJIA 2 2025      | pret casa               | FE (drop useless), ensemble model                                   | ML     |
| Simulare OJIA 3 2025      | admitere liceu          | FE (drop useless)                                                   | ML     |
| Simulare OJIA 3 2025      | scor examen             | FE (drop useless, ordinal encoding, dummy encoding), fill missing   | ML     |
| BCS PreONIA 2025          | quadrilingual land      | text cleaning, tfidf, clustering                                    | NLP    |
| BCS PreONIA 2025          | reality filter          | CNN classifier                                                      | CV     |
| BCS Beginner 1 2025       | fitness level           | FE (missing, dummy, domain), robust scaler                          | ML     |
| BCS Beginner 1 2025       | powerlifting            | FE (cleaning, missing, dummy, poly)                                 | ML     |
| BCS Intermediate 1 2025   | are you a robot         | count syllables, kmeans on tfidf                                    | NLP    |
| BCS Intermediate 1 2025   | autovalue car prices    | work in progress                                                    | ML     |
| BCS Intermediate 1 2025   | smart waste classifier  | cnn for classification                                              | CV     |
| Spooky Round 2025         | haunt me                | faster-rcnn finetune for obj detection                              | CV     |
| Spooky Round 2025         | creepy pizza            | lstm for sound classification                                       | Audio  |
| BCS Training Special 2025 | mnemon-dream arch       | modality encoders + autoregressive text gen (too much code)         | CV+NLP |
| BCS Training Special 2025 | grid based pathfinding  | work in progress                                                    | ML     |
| BCS Training Special 2025 | code blue protocol      | work in progress                                                    | ML     |
| Algolymp Winter 2025      | the grinch incident     | work in progress                                                    | CV     |
| Algolymp Winter 2025      | save the Christmas      | work in progress                                                    | NLP    |
| Decebal Tech 2025         | sami                    | image processing, find similar objects (clahe, rift, flann matcher) | CV     |
| Decebal Tech 2025         | riki                    | work in progress                                                    | NLP    |
| BCS Beginner 2 2026       | gigel and llms          | work in progress                                                    | NLP    |
| BCS Beginner 2 2026       | gigel and admission     | work in progress                                                    | ML     |
<!-- 
| Algolymp PreOJIA 9-10 2026  |                         |                                                                     |        |
| Algolymp PreOJIA 9-10 2026  |                         |                                                                     |        |
| Algolymp PreOJIA 11-12 2026 |                         |                                                                     |        |
| Algolymp PreOJIA 11-12 2026 |                         |                                                                     |        |
| CEOAI Practice Round 1 2026 |                         |                                                                     |        |
| CEOAI Practice Round 1 2026 |                         |                                                                     |        |
| CEOAI Practice Round 1 2026 |                         |                                                                     |        | -->

## MLCompete

Contests:

| competition                 | task                 | core idea                       | type |
| --------------------------- | -------------------- | ------------------------------- | ---- |
| ONIA Winter Warmup 2025     | rabbit exhibition    | df cleaning, grad boosting      | ML   |
| ONIA Winter Warmup 2025     | magic of words       | finetune clip to select words   | NLP  |
| ONIA Winter Warmup 2025     | glitch hunter        | binary masking using unet       | CV   |
| Local Stage Simulation 2026 | paintings            | FE (cluster features, cleaning) | ML   |
| Local Stage Simulation 2026 | parkinson            | grad boosting, auc              | ML   |
| Local Stage Simulation 2026 | transport            | clustering                      | ML   |
| "Unirea" College Round 2026 | bac under scrutiny   | time series, lagging features   | ML   |
| "Unirea" College Round 2026 | petronel the cyclist | work in progress                | ML   |

Most interesting/useful tasks:

| task name                                                                          | core idea                 | type   |
| ---------------------------------------------------------------------------------- | ------------------------- | ------ |
| [Text correction](https://platform.olimpiada-ai.ro/en/problems/46)                 | t5 finetune               | NLP    |
| [Residency Exam](https://platform.olimpiada-ai.ro/en/problems/43)                  | vllm inference            | NLP    |
| [Similarity Oracle](https://platform.olimpiada-ai.ro/en/problems/39)               | bert finetune             | NLP    |
| [Emoji Segmentation](https://platform.olimpiada-ai.ro/en/problems/40)              | image processing          | CV     |
| [Counting Emojis](https://platform.olimpiada-ai.ro/en/problems/37)                 | image processing          | CV     |
| [The archive of a thousand rooms](https://platform.olimpiada-ai.ro/en/problems/57) | vqa, multimodality        | CV+NLP |
| [Saving Christmas](https://platform.olimpiada-ai.ro/en/problems/66)                | cnn regression            | CV     |
| [Pokémon – Professor Oak's Call](https://platform.olimpiada-ai.ro/en/problems/56)  | semi ssl, unbalanced data | CV     |

For a full list of what I have done (mostly easy classic ML), check the [mlcompete folder](./contests/mlcompete).
