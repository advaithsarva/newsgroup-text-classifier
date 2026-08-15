"""Regression tests. Each one names an old bug it exists to prevent.

Run:  python tests/test_pipeline.py

No pytest, no fixtures, no network, no dataset download — the whole suite runs
on a hand-written 6-document corpus. Stages you have not implemented yet are
reported as SKIP, not failure, so this is runnable from the first stage on.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import pipeline as P

# Deliberately includes what the old cleaner destroyed: capitalized words
# (NASA, Shuttle), numbers (1969, 2), a doubled letter (committee), and noise
# that should genuinely go (an email, a URL, a tag).
CORPUS = [
    "NASA launched the Shuttle in 1969. Contact nasa@example.com for details.",
    "The graphics card renders 2 million polygons, see http://example.com/gpu",
    "My committee discussed the budget at length <b>yesterday</b>.",
    "Orbital mechanics and rocket propulsion are space topics.",
    "This video card driver keeps crashing on boot.",
    "The committee vote was postponed again.",
]
TRUTH = [0, 1, 2, 0, 1, 2]

_results = []


def check(name, fn):
    try:
        fn()
    except NotImplementedError:
        _results.append(("SKIP", name, "not implemented yet"))
    except AssertionError as e:
        _results.append(("FAIL", name, str(e)))
    except Exception as e:
        _results.append(("FAIL", name, f"{type(e).__name__}: {e}"))
    else:
        _results.append(("PASS", name, ""))


# -- clean ----------------------------------------------------------------

def test_clean_returns_a_string():
    """Old bug: CleanText(files_path, output_file, cleaned_output_file) took
    three paths and wrote files, but was called as .apply(CleanText)."""
    out = P.clean(CORPUS[0])
    assert isinstance(out, str), f"clean() returned {type(out).__name__}, want str"


def test_clean_never_empties_a_document():
    """Old bug: 27 stacked regexes stripped so much that content vanished."""
    for doc in CORPUS:
        out = P.clean(doc)
        assert out.strip(), f"clean() emptied a normal document: {doc!r}"


def test_clean_keeps_content_words():
    """Old bug: r'\\b[A-Z][a-z]*\\b' deleted every capitalized word, so all
    proper nouns — the most discriminative terms in newsgroup text — went."""
    out = P.clean("NASA launched the Shuttle").lower()
    assert "nasa" in out, "capitalized words are being deleted"
    assert "shuttle" in out, "capitalized words are being deleted"


def test_clean_does_not_stem_into_nonwords():
    """Old bug: lemmatize(stem(w)) — stemming first destroys the form the
    lemmatizer needs. In the real output, 'economy' came out as 'eco nomi'.
    Whatever you do, words must stay whole and recognisable."""
    out = P.clean("The economy improved").lower()
    assert "economi" not in out.replace("economy", ""), "words are being stemmed to non-words"
    assert "economy" in out, "'economy' did not survive cleaning intact"


def test_clean_strips_noise():
    out = P.clean(CORPUS[0]).lower()
    assert "@" not in out, "email address survived cleaning"


def test_corpus_length_is_preserved():
    """THE bug. Cleaning collapsed the corpus into a single string, so
    TF-IDF saw one document and IDF was log(1/1) = 0 for every term."""
    out = P.clean_corpus(CORPUS)
    assert isinstance(out, list), f"clean_corpus returned {type(out).__name__}"
    assert len(out) == len(CORPUS), f"corpus went from {len(CORPUS)} docs to {len(out)}"


# -- vectorize ------------------------------------------------------------

def test_matrix_has_one_row_per_document():
    matrix, _ = P.vectorize(P.clean_corpus(CORPUS))
    assert matrix.shape[0] == len(CORPUS), (
        f"matrix has {matrix.shape[0]} rows for {len(CORPUS)} documents"
    )


def test_idf_actually_discriminates():
    """On a 1-document corpus every term has df == N, so IDF is 0 and TF-IDF
    degenerates to term frequency. Distinct terms must get distinct weights."""
    matrix, _ = P.vectorize(P.clean_corpus(CORPUS))
    assert matrix.shape[1] > 1, "vocabulary collapsed to one term"
    assert len(set(matrix.toarray().round(6).ravel())) > 2, "all weights identical"


# -- cluster --------------------------------------------------------------

def test_one_label_per_document():
    """Old bug: true_k = min(5, n_samples) silently became k=1."""
    matrix, _ = P.vectorize(P.clean_corpus(CORPUS))
    labels, _ = P.cluster(matrix, 3)
    assert len(labels) == len(CORPUS), f"{len(labels)} labels for {len(CORPUS)} docs"
    assert len(set(labels)) > 1, "everything landed in one cluster"


def test_clustering_beats_random():
    """The old project never scored its clustering at all, so a degenerate
    result was indistinguishable from a good one. ARI ~0.0 is random."""
    matrix, _ = P.vectorize(P.clean_corpus(CORPUS))
    labels, _ = P.cluster(matrix, 3)
    ari = P.score_clusters(TRUTH, labels)["ari"]
    assert ari > 0.0, f"clustering is no better than random (ARI {ari:.3f})"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            check(name[5:].replace("_", " "), fn)

    width = max(len(n) for _, n, _ in _results)
    for status, name, detail in _results:
        print(f"{status}  {name:<{width}}  {detail}")

    failed = sum(1 for s, _, _ in _results if s == "FAIL")
    skipped = sum(1 for s, _, _ in _results if s == "SKIP")
    passed = len(_results) - failed - skipped
    print(f"\n{passed} passed, {failed} failed, {skipped} not yet implemented")
    sys.exit(1 if failed else 0)
