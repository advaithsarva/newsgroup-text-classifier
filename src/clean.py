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

    No stemming and no lemmatising. The old version ran lemmatize(stem(word)),
    and stemming first destroys the form the lemmatizer needs, which is how
    "economy" became "eco nomi". Stop words are handled by TfidfVectorizer.
    """
    for pattern in NOISE:
        doc = pattern.sub(" ", doc)
    doc = re.sub(r"[^a-z0-9\s]+", " ", doc.lower())
    return " ".join(doc.split())


def clean_corpus(docs):
    """Clean every document. 500 documents in, 500 documents out.

    This is the invariant the old pipeline broke: it joined everything into a
    single string, so TF-IDF saw one document and IDF collapsed to zero.
    """
    return [clean(d) for d in docs]
