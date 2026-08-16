"""Loading a corpus - either 20 Newsgroups or a folder of your own documents."""

import os

from sklearn.datasets import fetch_20newsgroups

from osmod import OSMod

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


def load_folder(directory, extensions=(".txt",)):
    """Load your own corpus: one document per file, labelled by subfolder.

    Expects the usual layout, where each subfolder is a class:

        corpus/politics/post1.txt
        corpus/sport/post2.txt

    Returns (docs, labels, category_names), the same shape as load(), so every
    downstream stage works unchanged.

    Files are read with errors="replace" because real text corpora contain
    broken encodings, and one bad byte should not stop the run.
    """
    paths = OSMod.filter_files_walk(directory, list(extensions))
    if not paths:
        raise ValueError(f"no {'/'.join(extensions)} files under {directory}")

    docs, names, labels = [], [], []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        if not text.strip():
            continue
        category = os.path.basename(os.path.dirname(path))
        if category not in names:
            names.append(category)
        docs.append(text)
        labels.append(names.index(category))

    if not docs:
        raise ValueError(f"every file under {directory} was empty")
    return docs, labels, names
