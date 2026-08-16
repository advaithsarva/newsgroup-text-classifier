"""Supervised classification of newsgroup posts.

Replaces the spaCy textcat model in the original project, which was trained on
IMDB movie reviews with labels positive/negative - the wrong task entirely.

A linear SVM over TF-IDF is the standard strong baseline for this dataset. It
trains in seconds on CPU, which matters more than a fractional gain from a
transformer when the point is a correct, reportable number.
"""

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import LinearSVC

import data
import tfidf
from clean import clean_corpus
from csvjson import CSVJSON

# Test set is the official 20 Newsgroups by-date split: posts written after the
# training posts. Harder than a random split, and the honest one, because a
# random split lets near-duplicate replies to the same thread land on both
# sides and inflate the score.


def train(categories=None, min_df=2):
    """Train on the train split, evaluate on the held-out test split.

    Returns (metrics, model, vectorizer, report_text).
    """
    train_docs, y_train, names = data.load(categories, subset="train")
    test_docs, y_test, _ = data.load(categories, subset="test")

    X_train, vec = tfidf.vectorize(clean_corpus(train_docs), min_df=min_df)
    X_test = vec.transform(clean_corpus(test_docs))

    model = LinearSVC().fit(X_train, y_train)
    predicted = model.predict(X_test)

    metrics = {
        "train_docs": len(train_docs),
        "test_docs": len(test_docs),
        "categories": len(names),
        "features": X_train.shape[1],
        "accuracy": accuracy_score(y_test, predicted),
        "macro_f1": f1_score(y_test, predicted, average="macro"),
    }
    report = classification_report(y_test, predicted, target_names=names, digits=3)
    per_class = classification_report(
        y_test, predicted, target_names=names, output_dict=True
    )
    return metrics, per_class, report


if __name__ == "__main__":
    import argparse
    import os

    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="all 20 categories (default 4)")
    p.add_argument("--save", metavar="DIR", help="write metrics as JSON and CSV")
    args = p.parse_args()

    metrics, per_class, report = train(None if args.all else data.SUBSET_4)
    print(
        f"{metrics['train_docs']} train / {metrics['test_docs']} test documents, "
        f"{metrics['categories']} categories, {metrics['features']} features"
    )
    print(f"accuracy {metrics['accuracy']:.3f}   macro F1 {metrics['macro_f1']:.3f}\n")
    print(report)

    if args.save:
        CSVJSON.write_json(os.path.join(args.save, "classification.json"), metrics)
        CSVJSON.write_csv(
            os.path.join(args.save, "per_class.csv"),
            [
                {
                    "category": name,
                    "precision": round(s["precision"], 4),
                    "recall": round(s["recall"], 4),
                    "f1": round(s["f1-score"], 4),
                    "support": int(s["support"]),
                }
                for name, s in per_class.items()
                if isinstance(s, dict) and name != "accuracy"
            ],
        )
        print(f"saved to {args.save}/")
