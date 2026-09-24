"""Scatter plot of document clusters. Only module that imports matplotlib."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD


def plot_clusters(matrix, labels_pred, path, seed=42):
    """2D scatter of every document (TF-IDF reduced via TruncatedSVD),
    colored by predicted cluster. Shows the clustering itself, not just
    the top keywords of each cluster.
    """
    coords = TruncatedSVD(n_components=2, random_state=seed).fit_transform(matrix)
    labels_pred = np.asarray(labels_pred)
    clusters = sorted(set(labels_pred))
    cmap = plt.get_cmap("tab20", len(clusters))

    fig, ax = plt.subplots(figsize=(8, 6))
    for color_i, cluster_id in enumerate(clusters):
        mask = labels_pred == cluster_id
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=8,
            alpha=0.6,
            color=cmap(color_i),
            label=str(cluster_id),
        )
    ax.set_xlabel("SVD component 1")
    ax.set_ylabel("SVD component 2")
    ax.set_title(f"Document clusters (k={len(clusters)})")
    legend = ax.legend(
        title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8
    )
    fig.add_artist(legend)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
