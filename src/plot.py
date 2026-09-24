"""Interactive cluster scatter plot. Only module that imports plotly."""

import plotly.graph_objects as go
from sklearn.decomposition import TruncatedSVD


def plot_clusters(matrix, labels_pred, cluster_terms, path, seed=42):
    """Self-contained HTML scatter of every document (TF-IDF reduced via
    TruncatedSVD via 2D), colored by predicted cluster. Hovering a point
    shows its cluster id and that cluster's top topic words.
    """
    coords = TruncatedSVD(n_components=2, random_state=seed).fit_transform(matrix)
    clusters = sorted(set(labels_pred))

    fig = go.Figure()
    for cluster_id in clusters:
        mask = [label == cluster_id for label in labels_pred]
        xs = coords[mask, 0]
        ys = coords[mask, 1]
        topic = ", ".join(cluster_terms[cluster_id])
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=f"cluster {cluster_id}",
                marker=dict(size=6, opacity=0.6),
                text=[f"cluster {cluster_id}<br>topic: {topic}"] * len(xs),
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=f"Document clusters (k={len(clusters)})",
        xaxis_title="SVD component 1",
        yaxis_title="SVD component 2",
        legend_title="cluster",
    )
    fig.write_html(path)
