# Project SB — 20 Newsgroups text classification and topic modelling

Classifies and clusters Usenet posts across the 20 Newsgroups corpus:
supervised classification with TF-IDF and a linear SVM, unsupervised KMeans
clustering scored against the true labels, and LDA topic modelling.

**Two dependencies: scikit-learn and numpy.** No gensim, no NLTK, no spaCy.

| Task | Result |
|---|---|
| Classification, 20 categories | accuracy **0.706**, macro F1 **0.694** |
| Classification, 4 categories | accuracy **0.843**, macro F1 **0.840** |
| KMeans, 20 categories | ARI **0.112**, NMI **0.377** |
| KMeans, 4 categories | ARI **0.330**, NMI **0.455** |

Measured on the official by-date test split with headers, footers and quotes
removed. Full numbers, per-class breakdown and error analysis in
**[RESULTS.md](RESULTS.md)**.

## Quick start

```bash
pip install -r requirements.txt

python tests/test_pipeline.py      # 14 tests, offline, no downloads
python src/classify.py --all       # train and evaluate the classifier
python src/run.py --all            # cluster and score against true labels
python src/run.py --all --topics 20
```

The corpus downloads on first run (~14MB) and caches. Everything is seeded, so
results reproduce exactly.

## Layout

```
src/
  data.py       load 20 Newsgroups, strip label-leaking headers
  clean.py      normalise text
  tfidf.py      TF-IDF vectorisation
  kmeans.py     clustering, ARI/NMI scoring, top terms
  lda.py        bag-of-words counts and LDA topics
  classify.py   TF-IDF -> LinearSVC, train/test evaluation
  run.py        CLI, wires the stages together
tests/
  test_pipeline.py
notebooks/original/
  the 2023 notebooks, kept as a record - see below
```

Each module maps to one stage. Every stage takes and returns **a list with one
string per document** — the invariant the original code broke.

## Design decisions

**Headers, footers and quotes are removed at load.** The `Newsgroups:` header
names the target class, so leaving it in yields accuracy above 0.95 that
measures nothing. Every number here is on the harder, honest task.

**The by-date split is used, not a random one.** A random split lets
near-duplicate replies from the same thread appear in both train and test.

**No stemming and no lemmatising.** The original ran `lemmatize(stem(word))`;
stemming first destroys the form the lemmatizer needs, which turned "economy"
into `eco nomi`. Stop words are handled by `TfidfVectorizer(stop_words=...)`.

**LDA is fitted on counts, not TF-IDF weights**, because it is a generative
model over word counts.

**`cluster()` raises rather than clamping `k`.** See below.

## What the 2023 notebooks got wrong

`notebooks/original/` holds the first version of this project. Its outputs were
invalid, and it is kept because the failure is instructive. Everything below
was confirmed against the notebooks' own saved cell outputs.

**The corpus collapsed to a single document.** `CleanText` wrote the whole
corpus into a text file as one line; `TFIDF.ipynb` read it back with
`readlines()` and got a list of length 1. Three consequences:

- **TF-IDF broke.** IDF is `log(N / df(t))`. With one document that is
  `log(1/1) = 0` for every term, so the IDF half multiplied everything by zero.
- **KMeans broke.** `true_k = min(5, n_samples)` silently became `k=1`. The
  committed `outputcluster.txt` contains exactly one cluster. `cluster()` in
  `src/kmeans.py` now raises instead of clamping, so this cannot recur quietly.
- **PCA emitted `invalid value encountered in divide`**, which was saved in the
  notebook and never investigated.

**The pipeline never ran on 20 Newsgroups.** It was pointed at a group-project
PDF, a weather CSV and a `.docx`. The newsgroup files were never fed to it.

**Other confirmed defects:**

- `LoadTextDataToTextFile` had `return output_file` *inside* its `for` loop, so
  it processed exactly one file however many were passed
- its `.txt` branch used `readline()` — one line per file
- `CleanText` stacked 27 regexes including `\b[A-Z][a-z]*\b` (deletes every
  capitalized word, so all proper nouns) and `[-+]?\d*\.?\d+` (all numbers)
- `LDA.ipynb` built bigrams and TF-IDF-filtered the corpus in cells 8–9, then
  rebuilt `corpus` from the raw words in cell 10 and discarded all of it. It
  fit 30 topics on 21 documents
- the trained spaCy `textcat` artifact was an **IMDB sentiment classifier**
  (`positive`/`negative`, 500 examples) — the wrong task entirely. Replaced by
  `src/classify.py`
- a live Cohere API key was committed in the initial commit

None of this produced an error message. Everything ran, printed numbers and
wrote files.

## Tests

```bash
python tests/test_pipeline.py
```

14 tests, no pytest, no network, no dataset download — a six-document fixture
covers it. Each test names the specific historical bug it prevents, and the
suite was validated by running it against the original implementation, where 6
of them fail.

The sharpest one is not an assertion about output:

```python
def test_cluster_count_is_not_silently_clamped():
    # asking for more clusters than documents is a caller bug and must say so
```

The original bug was not a wrong number — it was a wrong number produced
silently.

## Removed from this project

All recoverable from git history:

- `TextModeling/Sources/` — the 20 newsgroup text dumps, superseded by
  `fetch_20newsgroups`; each file held a whole newsgroup, which is why the old
  code kept seeing 21 documents instead of ~18,000
- the IMDB spaCy model, its DocBins and configs, and the gensim LDA model
- `output/` — the invalid results described above
- `Task Scheduling/` — an unrelated Flask file-upload app
- `RAG.ipynb`, `Summarization.ipynb` — third-party tutorial notebooks, not
  original work. Summarization is no longer a goal
- `UI UX/` — an unrelated Loki series timeline demo
- duplicate and empty notebooks: `TFIDF-Copy1`, `TDIDF`, `Untitled1`,
  `Untitled2`
