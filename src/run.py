"""Run the pipeline end to end.

    python src/run.py                 4 categories, KMeans
    python src/run.py --all           all 20
    python src/run.py --all --topics 20
"""

import argparse

import data
import kmeans
import lda
import tfidf
from clean import clean_corpus


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="all 20 groups (default 4)")
    p.add_argument("-k", type=int, default=0, help="clusters (default: one per category)")
    p.add_argument("--topics", type=int, default=0, help="LDA topics, 0 to skip")
    args = p.parse_args(argv)

    docs, labels, names = data.load(None if args.all else data.SUBSET_4)
    print(f"{len(docs)} documents, {len(names)} categories")

    docs = clean_corpus(docs)
    assert len(docs) == len(labels), "cleaning changed the number of documents"

    matrix, vec = tfidf.vectorize(docs)
    print(f"tf-idf: {matrix.shape[0]} documents x {matrix.shape[1]} terms")

    labels_pred, model = kmeans.cluster(matrix, args.k or len(names))
    s = kmeans.score(labels, labels_pred)
    print(f"ARI {s['ari']:.3f}   NMI {s['nmi']:.3f}")
    for i, terms in enumerate(kmeans.top_terms(model, vec)):
        print(f"cluster {i}: {' '.join(terms)}")

    if args.topics:
        cmatrix, cvec = lda.counts(docs)
        for i, terms in enumerate(kmeans.top_terms(lda.topics(cmatrix, args.topics), cvec)):
            print(f"topic {i}: {' '.join(terms)}")


if __name__ == "__main__":
    main()
