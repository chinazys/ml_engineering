"""Dataset Exploration tab for the CIFAR-10 dashboard."""

import logging

import numpy as np
import streamlit as st

from src.data import CIFAR10_CLASSES, to_display_hwc
from src.viz import plot_class_distribution

logger = logging.getLogger(__name__)

_SPLIT_LABELS = {
    "Train": "train",
    "Validation": "val",
    "Test": "test",
}


def render(data_splits: dict) -> None:  # type: ignore[type-arg]
    """
    Render the Dataset Exploration tab.

    Args:
        data_splits: Dict with keys 'train', 'val', 'test', each a tuple (images, labels).
    """
    st.header("Dataset Exploration")

    st.subheader("Overview")
    train_imgs, train_lbls = data_splits["train"]
    val_imgs, val_lbls = data_splits["val"]
    test_imgs, test_lbls = data_splits["test"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Train samples", f"{len(train_lbls):,}")
    col2.metric("Validation samples", f"{len(val_lbls):,}")
    col3.metric("Test samples", f"{len(test_lbls):,}")
    col4.metric("Classes", str(len(CIFAR10_CLASSES)))

    st.markdown("**Class names:** " + ", ".join(CIFAR10_CLASSES))

    # Class distribution
    st.subheader("Class Distribution")
    split_choice = st.selectbox(
        "Select split for distribution",
        options=list(_SPLIT_LABELS.keys()),
        key="dist_split",
    )
    split_key = _SPLIT_LABELS[split_choice]
    _, labels = data_splits[split_key]
    st.plotly_chart(plot_class_distribution(labels, CIFAR10_CLASSES), use_container_width=True)

    st.divider()

    st.subheader("Sample Inspection")

    col_split, col_class = st.columns(2)
    with col_split:
        insp_split = st.selectbox(
            "Dataset split",
            options=list(_SPLIT_LABELS.keys()),
            key="insp_split",
        )
    split_key = _SPLIT_LABELS[insp_split]
    images, labels = data_splits[split_key]

    with col_class:
        class_filter = st.selectbox(
            "Filter by class (optional)",
            options=["All"] + CIFAR10_CLASSES,
            key="class_filter",
        )

    # Apply class filter
    if class_filter != "All":
        cls_idx = CIFAR10_CLASSES.index(class_filter)
        valid_indices = np.where(labels == cls_idx)[0]
    else:
        valid_indices = np.arange(len(labels))

    if len(valid_indices) == 0:
        st.warning("No samples found for the selected filter.")
        return

    sample_pos = st.slider(
        "Sample index (within filtered set)",
        min_value=0,
        max_value=int(len(valid_indices) - 1),
        value=0,
        key="sample_slider",
    )
    actual_idx = int(valid_indices[sample_pos])

    img = images[actual_idx]  # CHW float32
    label = int(labels[actual_idx])

    col_img, col_info = st.columns([1, 2])
    with col_img:
        display = to_display_hwc(img)
        st.image(display, caption=f"Sample #{actual_idx}", width=160)
    with col_info:
        st.markdown(f"**True label:** `{CIFAR10_CLASSES[label]}` (class {label})")
        st.markdown(f"**Dataset position:** {actual_idx}")
        st.markdown(f"**Image shape:** {img.shape[1]}×{img.shape[2]} px, {img.shape[0]} channels")
        st.markdown(f"**Pixel range:** [{img.min():.3f}, {img.max():.3f}]")

    logger.info("Displayed sample idx=%d label=%s split=%s", actual_idx, CIFAR10_CLASSES[label], split_key)
