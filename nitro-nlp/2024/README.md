# Nitro Language Processing - 3rd Edition - Satire

Binary classification of Romanian news articles into satire / non-satire buckets

https://www.kaggle.com/c/nitro-language-processing-3

Folosind transformers (0.90206):

- model preantrenat BERT (lr=1e-5)
- curatare extensiva a textului (vezi functia din dataset)
- possible improvement: qlora cu PEFT, antrenare pe mai mult din dataset (acum doar pe 0.1%)

First attempt (0.87183):

- se unesc title si content
- se foloseste un tfidf vectorizer cu ngram range default (1,1)
- logistic regression
- local un CV de 0.86659
