"""Tests. Each one exists to catch a bug the original code actually had.

    python tests/test_pipeline.py

Runs offline on the 6 documents below. No pytest, no downloads. Stages you
have not written yet show SKIP, so this is useful before the code exists.
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import clean as C
import data as D
import kmeans as K
import lda as L
import tfidf as T
from csvjson import CSVJSON
from osmod import OSMod

# Three topics, two documents each, with shared vocabulary inside each pair -
# otherwise there is nothing for clustering to find and the test proves nothing.
# Also contains what the old cleaner destroyed: capitalized words (NASA,
# Shuttle), numbers (1969), and noise that genuinely should go.
DOCS = [
    "NASA launched the Space Shuttle in 1969. Contact nasa@example.com.",
    "The graphics card renders 2 million polygons, see http://example.com/gpu",
    "My committee discussed the budget at length <b>yesterday</b>.",
    "Orbital mechanics and rocket propulsion are space flight topics.",
    "This video graphics card driver keeps crashing on boot.",
    "The committee vote on the budget was postponed again.",
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


def test_cluster_count_is_not_silently_clamped():
    # the old code did k = min(k, n_samples), turning an impossible request
    # into k=1 with no error. Asking for more clusters than documents is a
    # caller bug and must say so.
    m, _ = T.vectorize(C.clean_corpus(DOCS))
    try:
        K.cluster(m, len(DOCS) + 5)
    except ValueError:
        return
    raise AssertionError("k > n_documents was silently accepted")


# --- lda ---

def test_lda_uses_counts_not_weights():
    m, _ = L.counts(C.clean_corpus(DOCS))
    assert m.shape[0] == len(DOCS)
    assert m.dtype.kind in "iu", f"expected integer counts, got {m.dtype}"


def test_lda_gives_one_distribution_per_topic():
    m, vec = L.counts(C.clean_corpus(DOCS))
    model = L.topics(m, 3)
    assert model.components_.shape[0] == 3
    assert model.components_.shape[1] == len(vec.get_feature_names_out())
    assert len(K.top_terms(model, vec)) == 3


def test_lda_rejects_more_topics_than_documents():
    # the old notebook asked for 30 topics from 21 documents
    m, _ = L.counts(C.clean_corpus(DOCS))
    try:
        L.topics(m, len(DOCS) + 10)
    except ValueError:
        return
    raise AssertionError("more topics than documents was accepted")


# --- csvjson / osmod (from dsutil.py) ---

def test_json_round_trip():
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "nested", "metrics.json")
        payload = {"ari": 0.112, "categories": ["a", "b"], "documents": 6}
        CSVJSON.write_json(path, payload)
        assert CSVJSON.read_json(path) == payload
    finally:
        shutil.rmtree(tmp)


def test_csv_round_trip_infers_fieldnames():
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "per_class.csv")
        rows = [{"category": "sci.space", "f1": "0.76"}, {"category": "misc", "f1": "0.38"}]
        CSVJSON.write_csv(path, rows)  # fieldnames omitted on purpose
        assert CSVJSON.read_csv(path) == rows
    finally:
        shutil.rmtree(tmp)


def test_load_folder_labels_by_subfolder():
    tmp = tempfile.mkdtemp()
    try:
        for category, text in [("space", "orbit rocket launch"), ("guns", "firearm law")]:
            os.makedirs(os.path.join(tmp, category))
            for n in range(2):
                with open(os.path.join(tmp, category, f"{n}.txt"), "w") as f:
                    f.write(text)
        docs, labels, names = D.load_folder(tmp)
        assert len(docs) == 4, f"expected 4 documents, got {len(docs)}"
        assert len(labels) == len(docs)
        assert sorted(names) == ["guns", "space"]
        assert len(set(labels)) == 2
    finally:
        shutil.rmtree(tmp)


def test_load_folder_rejects_an_empty_directory():
    tmp = tempfile.mkdtemp()
    try:
        D.load_folder(tmp)
    except ValueError:
        return
    finally:
        shutil.rmtree(tmp)
    raise AssertionError("an empty directory was accepted")


def test_filter_files_walk_is_recursive_and_ordered():
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "deep", "deeper"))
        for name in ["b.txt", "a.txt", "skip.md"]:
            open(os.path.join(tmp, "deep", "deeper", name), "w").close()
        found = OSMod.filter_files_walk(tmp, [".txt"])
        assert len(found) == 2, f"expected 2 .txt files, got {len(found)}"
        assert found == sorted(found), "results must be deterministic"
    finally:
        shutil.rmtree(tmp)


def test_small_corpus_runs_end_to_end():
    # min_df was hardcoded to 5, which is right for 11,000 newsgroup posts and
    # empties the vocabulary on a small folder:
    #   ValueError: max_df corresponds to < documents than min_df
    import run

    tmp = tempfile.mkdtemp()
    out = tempfile.mkdtemp()
    try:
        words = {
            "space": "orbit rocket launch nasa shuttle satellite mission flight",
            "guns": "firearm gun rifle law control weapon amendment policy",
            "faith": "god church bible christian prayer belief worship spirit",
        }
        for category, text in words.items():
            os.makedirs(os.path.join(tmp, category))
            for n in range(3):
                with open(os.path.join(tmp, category, f"{n}.txt"), "w") as f:
                    f.write(f"{text} {n}")
        run.main(["--folder", tmp, "--topics", "3", "--save", out])
        saved = CSVJSON.read_json(os.path.join(out, "clustering.json"))
        assert saved["documents"] == 9
        assert saved["ari"] > 0.0
        assert os.path.exists(os.path.join(out, "topic_terms.csv"))
    finally:
        shutil.rmtree(tmp)
        shutil.rmtree(out)


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
