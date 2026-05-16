"""Visualization utilities for the CIFAR-10 dashboard."""

import logging
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def plot_class_distribution(labels: np.ndarray, class_names: List[str]) -> go.Figure:
    """Bar chart of class distribution."""
    counts = np.bincount(labels, minlength=len(class_names))
    fig = px.bar(
        x=class_names,
        y=counts.tolist(),
        labels={"x": "Class", "y": "Count"},
        title="Class Distribution",
        color=counts.tolist(),
        color_continuous_scale="Blues",
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    return fig


def plot_confusion_matrix(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_names: List[str],
) -> go.Figure:
    """Plotly confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix  # type: ignore[import-untyped]

    n = len(class_names)
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(n)))
    fig = px.imshow(
        cm,
        labels={"x": "Predicted", "y": "True", "color": "Count"},
        x=class_names,
        y=class_names,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
        text_auto=True,
    )
    fig.update_layout(width=600, height=550)
    return fig


def plot_per_class_errors(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_names: List[str],
) -> go.Figure:
    """Bar chart of per-class error counts."""
    errors = true_labels != pred_labels
    error_counts = np.zeros(len(class_names), dtype=int)
    for cls_idx in range(len(class_names)):
        mask = true_labels == cls_idx
        error_counts[cls_idx] = int((errors & mask).sum())

    fig = px.bar(
        x=class_names,
        y=error_counts.tolist(),
        labels={"x": "True Class", "y": "Error Count"},
        title="Per-Class Error Counts",
        color=error_counts.tolist(),
        color_continuous_scale="Reds",
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    return fig


def plot_probability_bar(
    probabilities: np.ndarray, class_names: List[str], predicted_idx: int
) -> go.Figure:
    """Horizontal bar chart of prediction probabilities."""
    colors = ["#EF553B" if i == predicted_idx else "#636EFA" for i in range(len(class_names))]
    fig = go.Figure(
        go.Bar(
            x=probabilities.tolist(),
            y=class_names,
            orientation="h",
            marker_color=colors,
        )
    )
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Probability",
        yaxis={"categoryorder": "total ascending"},
        height=350,
    )
    return fig


def fig_to_numpy(fig: plt.Figure) -> np.ndarray:  # type: ignore[name-defined]
    """Render a matplotlib Figure to a HWC uint8 numpy array."""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    import PIL.Image  # type: ignore[import-untyped]

    return np.array(PIL.Image.open(buf))
