"""Prediction & Explainability tab for the CIFAR-10 dashboard."""

import logging
from io import BytesIO
from typing import Optional

import numpy as np
import PIL.Image
import streamlit as st
import torch
import torch.nn.functional as F

from src.data import CIFAR10_CLASSES, normalize_images, to_display_hwc
from src.gradcam import GradCAM, overlay_heatmap
from src.mlflow_utils import list_experiments, list_runs, load_model_from_run
from src.viz import plot_probability_bar

logger = logging.getLogger(__name__)

_VALID_IMAGE_TYPES = ["png", "jpg", "jpeg"]


def _preprocess_uploaded(file_bytes: bytes) -> Optional[np.ndarray]:
    """Load an uploaded image and resize to 32x32 CHW float32 in [0,1]."""
    try:
        img = PIL.Image.open(BytesIO(file_bytes)).convert("RGB").resize((32, 32))
        arr = np.array(img, dtype=np.float32) / 255.0  # HWC
        return arr.transpose(2, 0, 1)  # CHW
    except Exception as exc:
        logger.error("Failed to decode uploaded image: %s", exc)
        return None


def _run_gradcam(model, image_chw_raw: np.ndarray, class_idx: Optional[int]) -> tuple:  # type: ignore[type-arg]
    """Normalize image and run Grad-CAM."""
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(3, 1, 1)
    image_norm = (image_chw_raw - mean) / std
    cam = GradCAM(model)
    heatmap, pred_cls, probs = cam.generate(image_norm, class_idx=class_idx)
    return heatmap, pred_cls, probs


def render(client, data_splits: dict, cfg: dict, device: torch.device) -> None:  # type: ignore[type-arg]
    """
    Render the Prediction & Explainability tab.

    Args:
        client: mlflow.MlflowClient instance.
        data_splits: Dict with 'train', 'val', 'test' splits (images, labels).
        cfg: Full config dict.
        device: Torch device.
    """
    st.header("Prediction & Explainability")

    st.subheader("Select MLflow Run")

    try:
        experiments = list_experiments(client)
    except Exception as exc:
        st.error(f"Cannot connect to MLflow: {exc}")
        return

    if not experiments:
        st.warning("No active experiments found.")
        return

    exp_names = [e["name"] for e in experiments]
    selected_exp_name = st.selectbox("Experiment", options=exp_names, key="exp_exp")
    selected_exp = next(e for e in experiments if e["name"] == selected_exp_name)

    try:
        runs = list_runs(client, selected_exp["experiment_id"])
    except Exception as exc:
        st.error(f"Failed to load runs: {exc}")
        return

    if not runs:
        st.warning("No runs found.")
        return

    run_names = [r["run_name"] for r in runs]
    selected_run_name = st.selectbox("Run", options=run_names, key="exp_run")
    selected_run = next(r for r in runs if r["run_name"] == selected_run_name)

    if st.button("Load Model", key="exp_load_btn"):
        with st.spinner("Loading model..."):
            model = load_model_from_run(
                client, selected_run["run_id"], cfg["model"]["n_classes"], device
            )
        if model is None:
            st.error("Could not load model artifact.")
            return
        st.session_state["exp_model"] = model
        st.success("Model loaded.")
        logger.info("Explainability model loaded for run %s", selected_run["run_id"])

    if "exp_model" not in st.session_state:
        st.info("Click 'Load Model' first.")
        return

    model = st.session_state["exp_model"]

    st.divider()

    st.subheader("Input Source")
    source = st.radio(
        "Choose input",
        options=["Dataset sample", "Upload image"],
        horizontal=True,
        key="exp_source",
    )

    image_raw: Optional[np.ndarray] = None  # CHW float32 [0,1], NOT normalized
    true_label: Optional[int] = None

    if source == "Dataset sample":
        split_choice = st.selectbox(
            "Split", options=["Train", "Validation", "Test"], key="exp_split"
        )
        split_key = {"Train": "train", "Validation": "val", "Test": "test"}[split_choice]
        images, labels = data_splits[split_key]

        class_filter = st.selectbox(
            "Filter by class (optional)",
            options=["All"] + CIFAR10_CLASSES,
            key="exp_class_filter",
        )
        if class_filter != "All":
            cls_idx = CIFAR10_CLASSES.index(class_filter)
            valid_indices = np.where(labels == cls_idx)[0]
        else:
            valid_indices = np.arange(len(labels))

        if len(valid_indices) == 0:
            st.warning("No samples for this filter.")
            return

        sample_pos = st.slider(
            "Sample index (within filtered set)",
            min_value=0,
            max_value=int(len(valid_indices) - 1),
            value=0,
            key="exp_sample_slider",
        )
        actual_idx = int(valid_indices[sample_pos])
        image_raw = images[actual_idx]
        true_label = int(labels[actual_idx])

    else:
        uploaded = st.file_uploader(
            "Upload an image (PNG / JPG, resized to 32×32)",
            type=_VALID_IMAGE_TYPES,
            key="exp_upload",
        )
        if uploaded is not None:
            file_bytes = uploaded.read()
            image_raw = _preprocess_uploaded(file_bytes)
            if image_raw is None:
                st.error("Could not decode image. Please upload a valid PNG or JPEG.")
                return
        else:
            st.info("Upload an image to continue.")
            return

    # Show the image
    st.image(to_display_hwc(image_raw), caption="Input image (32×32)", width=128)
    if true_label is not None:
        st.markdown(f"**True label:** `{CIFAR10_CLASSES[true_label]}`")

    st.divider()

    st.subheader("Inference")
    if st.button("Run Inference", key="exp_infer_btn"):
        mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(3, 1, 1)
        img_norm = ((image_raw - mean) / std).astype(np.float32)
        with torch.no_grad():
            tensor = torch.from_numpy(img_norm).unsqueeze(0).to(device)
            logits = model(tensor)
            single_probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_cls = int(np.argmax(single_probs))
        st.session_state["exp_probs"] = single_probs
        st.session_state["exp_pred_cls"] = pred_cls
        logger.info("Inference: predicted=%s conf=%.3f", CIFAR10_CLASSES[pred_cls], single_probs[pred_cls])

    if "exp_probs" in st.session_state:
        single_probs = st.session_state["exp_probs"]
        pred_cls = st.session_state["exp_pred_cls"]
        st.markdown(
            f"**Predicted:** `{CIFAR10_CLASSES[pred_cls]}` "
            f"— confidence **{single_probs[pred_cls]:.2%}**"
        )
        st.plotly_chart(
            plot_probability_bar(single_probs, CIFAR10_CLASSES, pred_cls),
            use_container_width=True,
        )

    st.divider()

    st.subheader("Grad-CAM Explanation")

    explain_class = st.selectbox(
        "Class to explain",
        options=["Top prediction"] + CIFAR10_CLASSES,
        key="exp_gradcam_class",
    )
    target_cls_idx: Optional[int] = None
    if explain_class != "Top prediction":
        target_cls_idx = CIFAR10_CLASSES.index(explain_class)

    if st.button("Generate Grad-CAM", key="exp_gradcam_btn"):
        with st.spinner("Computing Grad-CAM..."):
            try:
                heatmap, pred_cls, probs = _run_gradcam(model, image_raw, target_cls_idx)
                overlay = overlay_heatmap(image_raw, heatmap)
                st.session_state["exp_heatmap"] = heatmap
                st.session_state["exp_overlay"] = overlay
                st.session_state["exp_gradcam_pred"] = pred_cls
                st.session_state["exp_gradcam_probs"] = probs
                logger.info("Grad-CAM generated, predicted class=%s", CIFAR10_CLASSES[pred_cls])
            except Exception as exc:
                st.error(f"Grad-CAM failed: {exc}")
                logger.error("Grad-CAM error: %s", exc)
                return

    if "exp_overlay" in st.session_state:
        heatmap = st.session_state["exp_heatmap"]
        overlay = st.session_state["exp_overlay"]
        pred_cls = st.session_state["exp_gradcam_pred"]

        explained_cls = target_cls_idx if target_cls_idx is not None else pred_cls
        st.markdown(
            f"Explaining class: **`{CIFAR10_CLASSES[explained_cls]}`** "
            f"| Model predicts: **`{CIFAR10_CLASSES[pred_cls]}`**"
        )

        col_orig, col_heat, col_overlay = st.columns(3)
        with col_orig:
            st.image(to_display_hwc(image_raw), caption="Original", width=128)
        with col_heat:
            import matplotlib.cm as cm
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
            ax.axis("off")
            st.pyplot(fig, use_container_width=False)
            st.caption("Heatmap")
            plt.close(fig)
        with col_overlay:
            st.image(overlay, caption="Overlay", width=128)
