# Nitro Language Processing - 3rd Edition - Satire

Binary classification of Romanian news articles into satire / non-satire buckets

## Rezolvare: 0.87183

First attempt:

- se unesc title si content
- se foloseste un tfidf vectorizer cu ngram range default (1,1)
- logistic regression
- local un CV de 0.86659
