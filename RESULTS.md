# Results

All numbers below are reproducible with the commands shown. The dataset is
20 Newsgroups via `sklearn.datasets.fetch_20newsgroups`, loaded with
`remove=("headers", "footers", "quotes")`.

**Why that removal matters:** the `Newsgroups:` header names the target class
outright, and quoted reply text duplicates content across posts. Leaving them
in produces accuracy above 0.95 that measures nothing. Every figure here is on
the harder, honest version of the task, which is why they should be compared
against published `remove=`-enabled baselines and not against raw 20 Newsgroups
scores.

The train/test split is the official by-date split: test posts were written
after training posts. A random split would let near-duplicate replies from one
thread land on both sides and inflate the score.

---

## Supervised classification

`python src/classify.py --all`

TF-IDF → LinearSVC.

| Categories | Train | Test | Features | Accuracy | Macro F1 |
|---|---|---|---|---|---|
| 4 | 2,148 | 1,427 | 12,675 | **0.843** | **0.840** |
| 20 | 11,014 | 7,317 | 20,000 | **0.706** | **0.694** |

Best and worst of the 20 classes:

| Category | F1 |
|---|---|
| rec.sport.hockey | 0.876 |
| rec.sport.baseball | 0.803 |
| talk.politics.mideast | 0.794 |
| … | |
| alt.atheism | 0.520 |
| talk.politics.misc | 0.502 |
| talk.religion.misc | 0.377 |

The error pattern is the expected one and worth being able to explain: the
model is strong on categories with distinctive vocabulary (hockey, baseball,
crypto) and weak where categories overlap semantically.
`talk.religion.misc`, `alt.atheism` and `soc.religion.christian` discuss the
same subject matter and are separated mainly by stance, which bag-of-words
features cannot represent. `talk.religion.misc` is also the smallest class
(245 test documents).

---

## KMeans clustering

`python src/run.py --all`

Unsupervised. Labels are used only for scoring, never for fitting.

| Categories | Documents | Terms | ARI | NMI |
|---|---|---|---|---|
| 4 | 2,148 | 5,683 | **0.330** | **0.455** |
| 20 | 11,014 | 17,142 | **0.112** | **0.377** |

ARI is 0.0 for random assignment and 1.0 for a perfect match, so 0.112 is well
above chance but far from the supervised result — which is the honest and
expected outcome. Clustering never sees a label, and several newsgroups are not
separable by vocabulary alone.

ARI falls much further than NMI from 4 to 20 categories because ARI penalises
splitting one true class across several clusters, and that is exactly what
happens here: the five `comp.*` groups share so much vocabulary that KMeans
carves them along different lines than the newsgroup boundaries.

Clusters are nonetheless clearly interpretable, which is the real evidence the
pipeline works:

```
cluster  4: armenians armenian turkish turks armenia turkey genocide
cluster  7: israel israeli jews arab arabs jewish israelis lebanon peace
cluster 11: key encryption clipper chip keys escrow government secure nsa
cluster 13: team game games hockey season players play teams league win
cluster 15: drive drives hard disk floppy scsi cd boot internal problem
cluster 18: god jesus bible christians christian people believe christ faith
```

One cluster is an artifact worth noting rather than hiding:

```
cluster 10: chastity n3jxp shameful intellect skepticism surrender gordon banks
```

That is a single `sci.med` poster's signature block, repeated across enough
posts to form its own cluster. `remove=("footers",)` strips conventional
signatures but not this one. It is a good illustration that a clean-looking
clustering result can still contain a cluster that has learned an author rather
than a topic.

---

## LDA topic modelling

`python src/run.py --all --topics 20`

Fitted on bag-of-words counts, not TF-IDF weights, since LDA is a generative
model over counts.

Coherent topics:

```
topic  4: key government encryption president chip public use clipper security
topic  7: god jesus believe does bible mr faith christ christian say
topic  8: space nasa launch earth satellite ground shuttle orbit use wire
topic 15: armenian turkish armenians turkey people turks jews armenia genocide
topic 16: window use file server using set widget motif application windows
```

Two topics captured noise rather than meaning:

```
topic  9: 10 00 25 15 11 12 20 14 16 13
topic 13: ax max pl giz bhj 1t 34u wm 3t g9v
```

Topic 9 is bare numbers; topic 13 is the tail of a base64-encoded binary posted
to a newsgroup. Both are honest failures of a token filter that keeps
alphanumerics, and both are fixable with a minimum token length and a rule
against tokens mixing digits and letters. Left in place because the report
should show what the pipeline actually produced.

---

## Reproducing

```bash
pip install -r requirements.txt
python tests/test_pipeline.py      # 14 tests
python src/classify.py --all
python src/run.py --all --topics 20
```

The corpus downloads automatically on first run (~14MB). Every stage is seeded
with `random_state=42`, so these numbers reproduce exactly.
