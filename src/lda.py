"""LDA topic modelling. Replaces LDA.ipynb.

sklearn ships LatentDirichletAllocation, so gensim is not needed - which also
avoids gensim's numpy 2.x problems on Python 3.12.
"""

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def counts(docs, max_features=20000, min_df=1, max_df=0.5):
    """Bag-of-words counts. Return (matrix, vectorizer).

    LDA is a generative model over word counts, so it takes counts rather than
    the TF-IDF weights used for clustering.
    """
    v = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words="english",
    )
    return v.fit_transform(docs), v


def topics(matrix, n_topics, seed=42):
    """Fit LDA. Return the fitted model.

    The old notebook asked for 30 topics from 21 documents, so the guard
    matters. It also built bigrams and TF-IDF-filtered the corpus in cells
    8-9, then rebuilt it from the raw words in cell 10 and threw all of it
    away - which is why the pipeline here is linear and has no hidden rebuild.
    """
    if n_topics >= matrix.shape[0]:
        raise ValueError(f"{n_topics} topics from {matrix.shape[0]} documents")
    model = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        max_iter=20,
        random_state=seed,
    )
    return model.fit(matrix)
