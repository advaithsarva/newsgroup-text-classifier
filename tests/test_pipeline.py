"""Tests. Each one exists to catch a bug the original code actually had.

    python tests/test_pipeline.py

Runs offline on the 6 documents below. No pytest, no downloads. Stages you
have not written yet show SKIP, so this is useful before the code exists.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import clean as C
import kmeans as K
import tfidf as T

# Contains what the old cleaner destroyed: capitalized words (NASA, Shuttle),
# numbers (1969), and noise that genuinely should go.
DOCS = [
    "NASA launched the Shuttle in 1969. Contact nasa@example.com for details.",
    "The graphics card renders 2 million polygons, see http://example.com/gpu",
    "My committee discussed the budget at length <b>yesterday</b>.",
    "Orbital mechanics and rocket propulsion are space topics.",
    "This video card driver keeps crashing on boot.",
    "The committee vote was postponed again.",
]
TRUTH = [0, 1, 2, 0, 1, 2]


# --- clean ---

def test_returns_a_string():
    # old: CleanText took 3 file paths but was called with one string
    assert isinstance(C.clean(DOCS[0]), str)


def test_never_empties_a_document():
    # old: 27 stacked regexes stripped so much that content vanished
    for d in DOCS:
        assert C.clean(d).strip(), f"emptied: {d!r}"


def test_keeps_capitalized_words():
    # old: \b[A-Z][a-z]*\b deleted every proper noun
    out = C.clean("NASA launched the Shuttle").lower()
    assert "nasa" in out and "shuttle" in out


def test_does_not_stem_into_nonwords():
    # old: lemmatize(stem(w)) turned "economy" into "eco nomi"
    out = C.clean("The economy improved").lower()
    assert "economy" in out


def test_strips_noise():
    assert "@" not in C.clean(DOCS[0])


def test_corpus_keeps_its_length():
    # THE bug. Cleaning collapsed everything into one string.
    out = C.clean_corpus(DOCS)
    assert isinstance(out, list)
    assert len(out) == len(DOCS), f"{len(DOCS)} docs in, {len(out)} out"


# --- tfidf ---

def test_one_row_per_document():
    m, _ = T.vectorize(C.clean_corpus(DOCS))
    assert m.shape[0] == len(DOCS), f"{m.shape[0]} rows for {len(DOCS)} docs"


def test_idf_discriminates():
    # on a 1-document corpus IDF is log(1/1) = 0 and every weight is identical
    m, _ = T.vectorize(C.clean_corpus(DOCS))
    assert m.shape[1] > 1, "vocabulary collapsed"
    assert len(set(m.toarray().round(6).ravel())) > 2, "all weights identical"


# --- kmeans ---

def test_one_label_per_document():
    # old: true_k = min(5, n_samples) silently became k=1
    m, _ = T.vectorize(C.clean_corpus(DOCS))
    labels, _ = K.cluster(m, 3)
    assert len(labels) == len(DOCS)
    assert len(set(labels)) > 1, "everything landed in one cluster"


def test_clustering_beats_random():
    m, _ = T.vectorize(C.clean_corpus(DOCS))
    labels, _ = K.cluster(m, 3)
    ari = K.score(TRUTH, labels)["ari"]
    assert ari > 0.0, f"no better than random (ARI {ari:.3f})"


if __name__ == "__main__":
    results = []
    for name, fn in list(globals().items()):
        if not name.startswith("test_"):
            continue
        label = name[5:].replace("_", " ")
        try:
            fn()
        except NotImplementedError:
            results.append(("SKIP", label, "not written yet"))
        except AssertionError as e:
            results.append(("FAIL", label, str(e)))
        except Exception as e:
            results.append(("FAIL", label, f"{type(e).__name__}: {e}"))
        else:
            results.append(("PASS", label, ""))

    w = max(len(n) for _, n, _ in results)
    for status, name, detail in results:
        print(f"{status}  {name:<{w}}  {detail}")

    failed = sum(1 for s, _, _ in results if s == "FAIL")
    skipped = sum(1 for s, _, _ in results if s == "SKIP")
    print(f"\n{len(results) - failed - skipped} passed, {failed} failed, {skipped} not written yet")
    sys.exit(1 if failed else 0)
