"""Run the pipeline end to end.

    python src/run.py                          4 categories, KMeans
    python src/run.py --all                    all 20
    python src/run.py --all --topics 20
    python src/run.py --folder path/to/corpus  your own documents
    python src/run.py --all --save results     write JSON + CSV
"""

import argparse
import os

import data
import kmeans
import lda
import tfidf
from clean import clean_corpus
from csvjson import CSVJSON


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="all 20 groups (default 4)")
    p.add_argument("--folder", help="load your own corpus, one class per subfolder")
    p.add_argument("-k", type=int, default=0, help="clusters (default: one per category)")
    p.add_argument("--topics", type=int, default=0, help="LDA topics, 0 to skip")
    p.add_argument("--save", metavar="DIR", help="write results as JSON and CSV")
    p.add_argument(
        "--min-df",
        type=int,
        default=0,
        help="ignore terms in fewer than N documents (default: scaled to corpus size)",
    )
    args = p.parse_args(argv)

    if args.folder:
        docs, labels, names = data.load_folder(args.folder)
    else:
        docs, labels, names = data.load(None if args.all else data.SUBSET_4)
    print(f"{len(docs)} documents, {len(names)} categories")

    docs = clean_corpus(docs)
    assert len(docs) == len(labels), "cleaning changed the number of documents"

    # min_df has to scale with the corpus. A fixed 5 is right for 11,000
    # newsgroup posts and empties the vocabulary on a 9-document folder.
    min_df = args.min_df or (5 if len(docs) >= 500 else 1)

    matrix, vec = tfidf.vectorize(docs, min_df=min_df)
    print(f"tf-idf: {matrix.shape[0]} documents x {matrix.shape[1]} terms")

    labels_pred, model = kmeans.cluster(matrix, args.k or len(names))
    scores = kmeans.score(labels, labels_pred)
    print(f"ARI {scores['ari']:.3f}   NMI {scores['nmi']:.3f}")

    cluster_terms = kmeans.top_terms(model, vec)
    for i, terms in enumerate(cluster_terms):
        print(f"cluster {i}: {' '.join(terms)}")

    topic_terms = []
    if args.topics:
        cmatrix, cvec = lda.counts(docs, min_df=min_df)
        topic_terms = kmeans.top_terms(lda.topics(cmatrix, args.topics), cvec)
        for i, terms in enumerate(topic_terms):
            print(f"topic {i}: {' '.join(terms)}")

    if args.save:
        summary = {
            "documents": len(docs),
            "categories": names,
            "features": matrix.shape[1],
            "clusters": len(cluster_terms),
            "ari": scores["ari"],
            "nmi": scores["nmi"],
        }
        CSVJSON.write_json(os.path.join(args.save, "clustering.json"), summary)
        CSVJSON.write_csv(
            os.path.join(args.save, "cluster_terms.csv"),
            [{"cluster": i, "terms": " ".join(t)} for i, t in enumerate(cluster_terms)],
        )
        if topic_terms:
            CSVJSON.write_csv(
                os.path.join(args.save, "topic_terms.csv"),
                [{"topic": i, "terms": " ".join(t)} for i, t in enumerate(topic_terms)],
            )
        print(f"\nsaved to {args.save}/")


if __name__ == "__main__":
    main()
