"""TF-IDF vectorization. Replaces TFIDF.ipynb.

    TF(t,d)     how often term t appears in document d
    IDF(t)      log(N / df(t))  -  N documents, df(t) of them contain t
    TF-IDF      TF * IDF

The IDF half only works across many documents. On a one-document corpus every
term has df = N = 1, so IDF is log(1) = 0 and the whole thing collapses. That
is what happened before.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize(docs, max_features=20000, min_df=1, max_df=0.5):
    """Fit TF-IDF. Return (matrix, vectorizer), one row per document.

    max_features was 100 in the old notebook, far too small for 20 newsgroups.
    sublinear_tf dampens long posts that repeat a word many times.
    max_df drops terms in over half the corpus - they cannot discriminate.
    """
    v = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words="english",
        sublinear_tf=True,
    )
    return v.fit_transform(docs), v
