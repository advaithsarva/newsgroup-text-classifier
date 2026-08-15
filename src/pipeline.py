"""20 Newsgroups text pipeline: load -> clean -> vectorize -> cluster / topics.

THE CONTRACT, and the reason this file exists:

    A corpus is a list[str]. One string per document. It stays that way
    through every stage.

The old notebook version wrote the whole corpus into one .txt file, read it
back with readlines(), and got a list of length 1. TF-IDF on one document
gives IDF = log(1/1) = 0 for every term, and KMeans then "clustered" a single
sample into a single cluster. Every function below takes a list and returns a
list of the same length. tests/test_pipeline.py fails if that ever breaks.

Stages marked TODO are yours to fill in.
"""

from __future__ import annotations

import argparse
import re

# Categories are the 20 Newsgroups labels. Passing None to load_newsgroups
# uses all 20; pass a subset while iterating so runs stay fast.
SUBSET_4 = [
    "alt.atheism",
    "comp.graphics",
    "sci.space",
    "talk.politics.guns",
]


# --------------------------------------------------------------------------
# Load  (done — no reason to hand-parse the .txt files in Sources/)
# --------------------------------------------------------------------------

def load_newsgroups(categories=None, subset="train", seed=42):
    """Return (docs, labels, label_names).

    docs is list[str] — one post per element. This replaces the whole
    LoadTextDataToTextFile / CheckFileType path from Preprocessing.ipynb.

    headers/footers/quotes are stripped because they leak the answer: the
    'Newsgroups:' header literally names the target class, and any classifier
    trained with them reports a fake ~99% accuracy.
    """
    from sklearn.datasets import fetch_20newsgroups

    bunch = fetch_20newsgroups(
        subset=subset,
        categories=categories,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=seed,
    )
    docs = [d for d in bunch.data if d.strip()]
    labels = [l for d, l in zip(bunch.data, bunch.target) if d.strip()]
    return docs, labels, list(bunch.target_names)


# --------------------------------------------------------------------------
# Clean  (TODO — this is the stage that was destroying the corpus)
# --------------------------------------------------------------------------

# Kept from the old CleanText because these are the patterns that were
# actually worth removing. The 20-odd others deleted real content:
#   r'\b[A-Z][a-z]*\b'   killed every capitalized word
#   r'[-+]?\d*\.?\d+'    killed every number
#   r'(.)\1{2,}'         mangled any word with a doubled letter
#   the "phone number" pattern matched nearly any run of digits
# Add back only what you can justify, and add a test for each one.
NOISE = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),        # email addresses
    re.compile(r"https?://\S+"),                    # URLs
    re.compile(r"<[^>]+>"),                         # HTML/markup tags
]


def clean(doc: str) -> str:
    """Clean ONE document. Returns a string, never None, never ''.

    TODO(you): implement.

    Rules that the tests enforce:
      - takes one str, returns one str. Do not take file paths, do not write
        files. The old signature was CleanText(files_path, output_file,
        cleaned_output_file) but it was called as df["text"].apply(CleanText).
      - do not join documents together. Cleaning is per-document.
      - lowercase, strip NOISE, collapse whitespace is enough to start.
      - do NOT stem and lemmatize the same token. The old code ran
        lemmatize(stem(w)); stemming first destroys the word form the
        lemmatizer needs ("economy" -> "economi" -> "economi").
        Pick one. TfidfVectorizer's built-in stop_words='english' means you
        probably need neither.
    """
    raise NotImplementedError


def clean_corpus(docs: list[str]) -> list[str]:
    """Clean every document, preserving corpus length."""
    return [clean(d) for d in docs]


# --------------------------------------------------------------------------
# Vectorize  (TODO)
# --------------------------------------------------------------------------

def vectorize(docs: list[str], max_features: int = 20_000):
    """Return (matrix, vectorizer). matrix.shape[0] must equal len(docs).

    TODO(you): implement with sklearn's TfidfVectorizer.

    Worth deciding explicitly, since the old notebook set these blind:
      - max_features: old value was 100, which is far too small for 20 groups
      - min_df / max_df: old min_df=0.01 on a 1-document corpus meant nothing
      - stop_words="english" is built in; that is why nltk is not a dependency
      - sublinear_tf=True is usually a win on newsgroup-length text
    """
    raise NotImplementedError


# --------------------------------------------------------------------------
# Cluster  (TODO)
# --------------------------------------------------------------------------

def cluster(matrix, k: int, seed: int = 42):
    """Return (labels, model). labels has one cluster id per document,
    so len(labels) == matrix.shape[0]. The model is returned so top_terms
    can read its centroids.

    TODO(you): implement with sklearn's KMeans.

    The old code set true_k = min(5, n_samples), which on a 1-document corpus
    silently became k=1. Do not clamp k to the sample count; if k > n_samples
    that is a bug in the caller and should raise.
    """
    raise NotImplementedError


def score_clusters(true_labels, predicted):
    """How good is the clustering, against the real newsgroup labels?

    This is the number the old project never computed — it wrote the top terms
    per cluster to a text file and stopped, so there was no way to tell a good
    clustering from a random one.

    Adjusted Rand Index: 1.0 perfect, ~0.0 random. Normalized Mutual
    Information is reported alongside because ARI punishes splitting a true
    class across clusters harder than NMI does.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    return {
        "ari": adjusted_rand_score(true_labels, predicted),
        "nmi": normalized_mutual_info_score(true_labels, predicted),
    }


def top_terms(model, vectorizer, n: int = 10) -> list[list[str]]:
    """Top n terms per cluster/topic, for eyeballing what each one captured."""
    names = vectorizer.get_feature_names_out()
    centers = getattr(model, "cluster_centers_", None)
    if centers is None:
        centers = model.components_
    return [[names[i] for i in row.argsort()[::-1][:n]] for row in centers]


# --------------------------------------------------------------------------
# Topics  (TODO)
# --------------------------------------------------------------------------

def topics(matrix, n_topics: int, seed: int = 42):
    """Fit LDA. Return the fitted model.

    TODO(you): implement with sklearn's LatentDirichletAllocation.
    sklearn ships it, so gensim is not a dependency — which also sidesteps
    gensim's numpy 2.x incompatibility on Python 3.12.

    Two things the LDA notebook got wrong, worth not repeating:
      - it built bigrams and did TF-IDF filtering, then overwrote `corpus`
        from the raw words two cells later, discarding all of it
      - it fit 30 topics on ~21 documents. Keep n_topics well under the
        document count; near the number of true categories is a sane start.

    Note LDA wants raw counts, not TF-IDF weights. Either pass a
    CountVectorizer matrix here or accept that you are fitting on TF-IDF and
    say so.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--all", action="store_true", help="all 20 groups (default: 4)")
    p.add_argument("-k", type=int, default=0, help="clusters (default: #categories)")
    p.add_argument("--topics", type=int, default=0, help="LDA topics (0 = skip)")
    args = p.parse_args(argv)

    cats = None if args.all else SUBSET_4
    docs, labels, names = load_newsgroups(cats)
    print(f"{len(docs)} documents, {len(names)} categories")

    docs = clean_corpus(docs)
    assert len(docs) == len(labels), "cleaning changed the corpus length"

    matrix, vec = vectorize(docs)
    print(f"tf-idf matrix {matrix.shape[0]} docs x {matrix.shape[1]} terms")

    k = args.k or len(names)
    predicted, km = cluster(matrix, k)
    scores = score_clusters(labels, predicted)
    print(f"ARI {scores['ari']:.3f}   NMI {scores['nmi']:.3f}")
    for i, terms in enumerate(top_terms(km, vec)):
        print(f"cluster {i}: {' '.join(terms)}")

    if args.topics:
        lda = topics(matrix, args.topics)
        for i, terms in enumerate(top_terms(lda, vec)):
            print(f"topic {i}: {' '.join(terms)}")


if __name__ == "__main__":
    main()
