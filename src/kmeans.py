"""KMeans clustering and scoring. Replaces the cluster cells in TFIDF.ipynb."""

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def cluster(matrix, k, seed=42):
    """Fit KMeans. Return (labels, model) - one label per document.

    TODO: implement.

    Do not write k = min(k, n_samples). The old code did, and on a
    one-document corpus that silently became k=1, which is why
    outputcluster.txt has a single cluster. If k > n_samples the caller is
    wrong and it should fail loudly.
    """
    raise NotImplementedError


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
