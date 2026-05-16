"""Error Analysis tab for the CIFAR-10 dashboard."""

import logging
from typing import Optional

import numpy as np
import streamlit as st

from src.data import CIFAR10_CLASSES, to_display_hwc
from src.inference import find_misclassified, run_inference
from src.mlflow_utils import list_experiments, list_runs, load_model_from_run
from src.viz import plot_confusion_matrix, plot_per_class_errors

logger = logging.getLogger(__name__)


def render(client, test_data: tuple, cfg: dict, device) -> None:  # type: ignore[type-arg]
    """
    Render the Error Analysis tab.

    Args:
        client: mlflow.MlflowClient instance.
        test_data: Tuple (images, labels) for the test set.
        cfg: Full config dict.
        device: Torch device.
    """
    st.header("Model Error Analysis")

    st.subheader("Select MLflow Run")

    try:
        experiments = list_experiments(client)
    except Exception as exc:
        st.error(f"Cannot connect to MLflow: {exc}")
        logger.error("MLflow connection error: %s", exc)
        return

    if not experiments:
        st.warning("No active experiments found in MLflow.")
        return

    exp_names = [e["name"] for e in experiments]
    selected_exp_name = st.selectbox("Experiment", options=exp_names, key="err_exp")
    selected_exp = next(e for e in experiments if e["name"] == selected_exp_name)

    try:
        runs = list_runs(client, selected_exp["experiment_id"])
    except Exception as exc:
        st.error(f"Failed to load runs: {exc}")
        logger.error("Failed to load runs: %s", exc)
        return

    if not runs:
        st.warning("No runs found for this experiment.")
        return

    run_names = [r["run_name"] for r in runs]
    selected_run_name = st.selectbox("Run", options=run_names, key="err_run")
    selected_run = next(r for r in runs if r["run_name"] == selected_run_name)

    # Show run metadata
    with st.expander("Run details"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Parameters**")
            for k, v in selected_run["params"].items():
                st.markdown(f"- `{k}`: {v}")
        with col2:
            st.markdown("**Metrics**")
            for k, v in selected_run["metrics"].items():
                st.markdown(f"- `{k}`: {v:.4f}" if isinstance(v, float) else f"- `{k}`: {v}")

    st.subheader("Error Analysis")

    if st.button("Run Error Analysis", key="run_analysis_btn"):
        run_id = selected_run["run_id"]
        with st.spinner("Loading model artifact from MLflow..."):
            model = load_model_from_run(client, run_id, cfg["model"]["n_classes"], device)

        if model is None:
            st.error("Could not load model artifact. Check MLflow artifacts for this run.")
            return

        test_images, test_labels = test_data
        with st.spinner("Running inference on test set..."):
            predictions, probabilities = run_inference(
                model,
                test_images,
                batch_size=cfg["app"]["inference_batch_size"],
                device=device,
            )

        # Store in session state so it persists
        st.session_state["err_predictions"] = predictions
        st.session_state["err_probabilities"] = probabilities
        st.session_state["err_test_labels"] = test_labels
        st.session_state["err_test_images"] = test_images
        logger.info("Error analysis complete for run %s", run_id)

    if "err_predictions" not in st.session_state:
        st.info("Click 'Run Error Analysis' to load the model and compute errors.")
        return

    predictions = st.session_state["err_predictions"]
    probabilities = st.session_state["err_probabilities"]
    test_labels = st.session_state["err_test_labels"]
    test_images = st.session_state["err_test_images"]

    accuracy = float((predictions == test_labels).mean())
    st.metric("Test accuracy", f"{accuracy:.4f}")

    # Confusion matrix
    st.subheader("Confusion Matrix")
    st.plotly_chart(
        plot_confusion_matrix(test_labels, predictions, CIFAR10_CLASSES),
        use_container_width=True,
    )

    # Per-class errors
    st.subheader("Per-Class Error Counts")
    st.plotly_chart(
        plot_per_class_errors(test_labels, predictions, CIFAR10_CLASSES),
        use_container_width=True,
    )

    # Individual misclassified examples
    st.subheader("Misclassified Samples")
    errors = find_misclassified(test_labels, predictions, probabilities)
    st.markdown(f"**Total misclassified:** {len(errors)} / {len(test_labels)}")

    # Sorting control
    sort_by = st.selectbox(
        "Sort errors by",
        options=["Confidence (high→low)", "Confidence (low→high)", "True class", "Predicted class"],
        key="err_sort",
    )
    if sort_by == "Confidence (high→low)":
        errors = sorted(errors, key=lambda x: x["confidence"], reverse=True)
    elif sort_by == "Confidence (low→high)":
        errors = sorted(errors, key=lambda x: x["confidence"])
    elif sort_by == "True class":
        errors = sorted(errors, key=lambda x: x["true_label"])
    else:
        errors = sorted(errors, key=lambda x: x["pred_label"])

    n_show = st.slider("Number of examples to display", min_value=1, max_value=min(50, len(errors)), value=12, key="err_n_show")
    cols_per_row = 4
    shown = errors[:n_show]

    for row_start in range(0, len(shown), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, err in enumerate(shown[row_start : row_start + cols_per_row]):
            with cols[col_idx]:
                img_display = to_display_hwc(test_images[err["index"]])
                st.image(img_display, width=80)
                st.caption(
                    f"True: **{CIFAR10_CLASSES[err['true_label']]}**\n"
                    f"Pred: {CIFAR10_CLASSES[err['pred_label']]}\n"
                    f"Conf: {err['confidence']:.2%}"
                )
