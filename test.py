from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)

X = vectorizer.transform(corpus)
print(vectorizer.vocabulary_)
print(X)
print(vectorizer.get_feature_names())
print(X.shape)
# X = np.array(X)
print(type(X))
print(np.array(X.mean(axis=1)))
