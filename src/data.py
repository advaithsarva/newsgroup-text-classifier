"""Loading the 20 Newsgroups corpus."""

from sklearn.datasets import fetch_20newsgroups

# A small subset to work with while iterating. Runs in seconds instead of minutes.
SUBSET_4 = ["alt.atheism", "comp.graphics", "sci.space", "talk.politics.guns"]


def load(categories=None, subset="train", seed=42):
    """Return (docs, labels, category_names). docs is a list of posts.

    This replaces reading Sources/*.txt by hand. Each of those files holds a
    whole newsgroup - thousands of posts in one file - which is why the old
    code kept ending up with 21 documents instead of ~18,000.

    headers/footers/quotes are removed because the 'Newsgroups:' header names
    the answer. Leave them in and you get ~99% accuracy that means nothing.
    """
    bunch = fetch_20newsgroups(
        subset=subset,
        categories=categories,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=seed,
    )
    pairs = [(d, l) for d, l in zip(bunch.data, bunch.target) if d.strip()]
    docs = [d for d, _ in pairs]
    labels = [l for _, l in pairs]
    return docs, labels, list(bunch.target_names)
