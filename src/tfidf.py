"""TF-IDF vectorization. Replaces TFIDF.ipynb.

    TF(t,d)     how often term t appears in document d
    IDF(t)      log(N / df(t))  -  N documents, df(t) of them contain t
    TF-IDF      TF * IDF

The IDF half only works across many documents. On a one-document corpus every
term has df = N = 1, so IDF is log(1) = 0 and the whole thing collapses. That
is what happened before.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize(docs, max_features=20000):
    """Fit TF-IDF. Return (matrix, vectorizer), matrix has one row per document.

    TODO: implement.

    Settings worth choosing deliberately, since the old notebook did not:
      max_features   was 100, far too small for 20 newsgroups
      min_df         drop terms appearing in fewer than N documents
      stop_words     "english" is built in, which is why nltk is not needed
      sublinear_tf   True usually helps on text this long
    """
    raise NotImplementedError
