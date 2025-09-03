Fake News Detecion from URLs:

This project levarages Machine Learning to detect fake news articles (via URLs or dataset).
It does so, by extracting titles from articles and applying TF-IDF vectorization and classifying them with an ensemble model(Logistic Regression, Naive Bayes, Random Forest).

This project is an end to end pipeline (data → preprocessing → training → evaluation).
It uses an ensemble model for robust classfification.

Dataset should be of this specified format:
`id, news_url, title, tweet_ids, label`
`label = 1 (real), 0 (fake)`