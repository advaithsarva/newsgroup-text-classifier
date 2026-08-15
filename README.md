# Project SB

Text classification and topic modeling on the 20 Newsgroups corpus —
cleaning, TF-IDF, KMeans clustering, LDA topics, and a spaCy text classifier.

Originally a set of research notebooks (2023). Being rebuilt as a runnable
pipeline; the notebooks are kept under `TextModeling/Code/` for reference.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The corpus is fetched by `sklearn.datasets.fetch_20newsgroups` on first run —
nothing to download by hand.

## Run

```bash
python src/pipeline.py                 # 4 categories, KMeans
python src/pipeline.py --all           # all 20
python src/pipeline.py --all --topics 20
```

## Tests

```bash
python tests/test_pipeline.py
```

No pytest, no network, no dataset download — a hand-written 6-document corpus
covers it. Every test names the original bug it exists to prevent. Stages not
yet implemented report `SKIP` rather than failing, so the suite is useful from
the first stage onward.

## Layout

| Path | What |
|---|---|
| `src/pipeline.py` | The pipeline. Load is done; clean/vectorize/cluster/topics are stubs. |
| `tests/test_pipeline.py` | Regression tests against the original bugs. |
| `TextModeling/Code/` | Original notebooks — reference only, see caveats below. |
| `TextModeling/Sources/` | Original corpus dumps. Superseded by the sklearn loader. |
| `Task Scheduling/` | Separate Flask file-upload app. Unrelated to the pipeline. |

## Notebook caveats

The notebooks under `TextModeling/Code/` produced the committed outputs, but
those outputs are not valid results. Known issues, kept here so nobody trusts
them by accident:

- `Preprocessing.ipynb` — `return output_file` sits inside the file loop, so
  only the first file is ever read. The `.txt` branch uses `readline()`, one
  line per file. It was run on a group-project PDF, a weather CSV and a
  `.docx` — the 20 Newsgroups files in `Sources/` were never fed to it.
- `CleanText` applies 27 stacked regexes that delete every capitalized word
  and every number, then runs `lemmatize(stem(w))`, which turns "economy" into
  "eco nomi". It writes the whole corpus as one line.
- `TFIDF.ipynb` — reads that one line with `readlines()`, giving a corpus of
  **one document**. IDF is `log(1/1) = 0` for every term, and
  `true_k = min(5, n_samples)` becomes k=1. `output/outputcluster.txt` is a
  single cluster.
- `LDA.ipynb` — builds bigrams and TF-IDF-filters the corpus, then overwrites
  `corpus` from the raw words two cells later, discarding all of it. Fits 30
  topics on ~21 documents. `pd` is never imported, so both CSVs fail, and the
  arxiv file is JSONL rather than JSON, so it fails too.
- `output/model-best` is an **IMDB sentiment classifier** (`positive` /
  `negative`, 500 training examples, macro-F 0.81) from `Untitled.ipynb` — not
  a newsgroup classifier. It also pins `spacy<3.8` and will not load on
  current spaCy.
- `Untitled2.ipynb` does not parse (`if match=True:`).

`Task Scheduling/ML model/RAG.ipynb` and `Summarization.ipynb` are third-party
tutorial notebooks, not original work.

## Configuration

Nothing is hardcoded to an absolute path any more.

| Variable | Used by | Default |
|---|---|---|
| `UPLOAD_FOLDER` | `Task Scheduling/app.py` | `Task Scheduling/Sources` |
| `FLASK_DEBUG` | `Task Scheduling/app.py` | off (`1` enables) |
| `COHERE_API_KEY` | `ML model/CodeSummarize.py` | required, no default |
