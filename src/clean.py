"""Text cleaning. Replaces CleanText from Preprocessing.ipynb."""

import re

# The three patterns from the old CleanText worth keeping. The other 24 deleted
# real content - \b[A-Z][a-z]*\b removed every capitalized word, and
# [-+]?\d*\.?\d+ removed every number. Add more only if you can justify each one.
NOISE = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    re.compile(r"https?://\S+"),
    re.compile(r"<[^>]+>"),
]


def clean(doc):
    """Clean one document. Takes a string, returns a string.

    TODO: implement.

    Lowercase, strip NOISE, collapse repeated whitespace is enough to start.

    Two things not to repeat from the old version:
      - it took three file paths and wrote files, but was called as
        df["text"].apply(CleanText) with a single string
      - it ran lemmatize(stem(word)). Stemming first destroys the form the
        lemmatizer needs, which is how "economy" became "eco nomi". Pick one,
        or neither - TfidfVectorizer has stop_words="english" built in.
    """
    raise NotImplementedError


def clean_corpus(docs):
    """Clean every document. 500 documents in, 500 documents out.

    This is the invariant the old pipeline broke: it joined everything into a
    single string, so TF-IDF saw one document and IDF collapsed to zero.
    """
    return [clean(d) for d in docs]
