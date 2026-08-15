"""KMeans clustering and scoring. Replaces the cluster cells in TFIDF.ipynb."""

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def cluster(matrix, k, seed=42):
    """Fit KMeans. Return (labels, model) - one label per document.

    Deliberately no k = min(k, n_samples). The old code clamped, and on a
    one-document corpus that silently became k=1 - which is why the original
    outputcluster.txt has a single cluster and no error. Fail loudly instead.
    """
    if k < 2:
        raise ValueError(f"need at least 2 clusters, got {k}")
    if k > matrix.shape[0]:
        raise ValueError(f"k={k} exceeds {matrix.shape[0]} documents")
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed)
    return model.fit_predict(matrix), model


def score(true_labels, predicted):
    """Compare clusters against the real newsgroup labels.

    The old project never did this - it wrote the top terms per cluster to a
    file and stopped, so there was no way to tell a good clustering from a
    broken one. It was broken.

    ARI: 1.0 is perfect, 0.0 is random guessing.
    NMI is reported too because ARI punishes splitting one true class across
    several clusters harder than NMI does.
    """
    return {
        "ari": adjusted_rand_score(true_labels, predicted),
        "nmi": normalized_mutual_info_score(true_labels, predicted),
    }


def top_terms(model, vectorizer, n=10):
    """Top n terms per cluster or topic, for eyeballing what each one caught."""
    names = vectorizer.get_feature_names_out()
    centers = getattr(model, "cluster_centers_", None)
    if centers is None:
        centers = model.components_
    return [[names[i] for i in row.argsort()[::-1][:n]] for row in centers]
