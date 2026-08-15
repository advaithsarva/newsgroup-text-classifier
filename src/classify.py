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
    return metrics, model, vec, report


if __name__ == "__main__":
    import sys

    cats = None if "--all" in sys.argv else data.SUBSET_4
    m, _, _, report = train(cats)
    print(
        f"{m['train_docs']} train / {m['test_docs']} test documents, "
        f"{m['categories']} categories, {m['features']} features"
    )
    print(f"accuracy {m['accuracy']:.3f}   macro F1 {m['macro_f1']:.3f}\n")
    print(report)
