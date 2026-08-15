"""LDA topic modelling. Replaces LDA.ipynb.

sklearn ships LatentDirichletAllocation, so gensim is not needed - which also
avoids gensim's numpy 2.x problems on Python 3.12.
"""

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def counts(docs, max_features=20000):
    """Bag-of-words counts. Return (matrix, vectorizer), same shape as tfidf.vectorize.

    LDA assumes raw counts, not TF-IDF weights.

    TODO: implement with CountVectorizer.
    """
    raise NotImplementedError


def topics(matrix, n_topics, seed=42):
    """Fit LDA. Return the fitted model.

    TODO: implement.

    Two things the old notebook did wrong:
      - cells 8-9 built bigrams and TF-IDF-filtered the corpus, then cell 10
        rebuilt `corpus` from the raw words and threw all of it away
      - it asked for 30 topics from 21 documents. Keep n_topics well below the
        document count. Near the number of real categories is a sane start.
    """
    raise NotImplementedError
