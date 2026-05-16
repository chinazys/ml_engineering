"""Streamlit dashboard for CIFAR-10 model analysis."""

import logging
import sys
from pathlib import Path

import mlflow
import streamlit as st
import torch
import yaml

_LAB_DIR = Path(__file__).parent
if str(_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(_LAB_DIR))

_REPO_ROOT = _LAB_DIR.parent.parent

from src.data import load_batches, load_test_set
from src.mlflow_utils import set_tracking_uri
from src.ui import dataset_tab, error_tab, explainability_tab

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_CONFIG_PATH = _LAB_DIR / "configs" / "config.yaml"


@st.cache_data(show_spinner=False)
def _load_config() -> dict:  # type: ignore[type-arg]
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    logger.info("Config loaded from %s", _CONFIG_PATH)
    return cfg


@st.cache_data(show_spinner="Loading dataset...")
def _load_data(data_dir: str, cifar_subdir: str, train_batches: list, val_batches: list, test_batch: str) -> dict:  # type: ignore[type-arg]
    full_data_dir = str(_REPO_ROOT / data_dir)
    train_images, train_labels = load_batches(full_data_dir, cifar_subdir, train_batches)
    val_images, val_labels = load_batches(full_data_dir, cifar_subdir, val_batches)
    test_images, test_labels = load_test_set(full_data_dir, cifar_subdir, test_batch)
    logger.info(
        "Dataset loaded: train=%d val=%d test=%d",
        len(train_labels), len(val_labels), len(test_labels),
    )
    return {
        "train": (train_images, train_labels),
        "val": (val_images, val_labels),
        "test": (test_images, test_labels),
    }


@st.cache_resource(show_spinner=False)
def _get_mlflow_client(tracking_uri: str) -> mlflow.MlflowClient:
    resolved_uri = tracking_uri
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        db_path = _REPO_ROOT / tracking_uri.replace("sqlite:///", "")
        resolved_uri = f"sqlite:///{db_path}"
    set_tracking_uri(resolved_uri)
    client = mlflow.MlflowClient(tracking_uri=resolved_uri)
    logger.info("MLflow client created with URI: %s", resolved_uri)
    return client


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    cfg = _load_config()

    st.set_page_config(
        page_title=cfg["app"]["page_title"],
        layout="wide",
    )
    st.title(cfg["app"]["page_title"])
    st.caption("Lab 06: Interactive Streamlit Dashboard for CIFAR-10 Model Analysis")

    device = _get_device()
    st.sidebar.markdown(f"**Device:** `{device}`")
    st.sidebar.markdown(f"**MLflow URI:** `{cfg['mlflow']['tracking_uri']}`")

    data_cfg = cfg["data"]
    data_splits = _load_data(
        data_cfg["data_dir"],
        data_cfg["cifar_subdir"],
        data_cfg["train_batches"],
        data_cfg["val_batches"],
        data_cfg["test_batch"],
    )

    mlflow_client = _get_mlflow_client(cfg["mlflow"]["tracking_uri"])

    tab_dataset, tab_errors, tab_explain = st.tabs(
        ["Dataset Exploration", "Error Analysis", "Prediction & Explainability"]
    )

    with tab_dataset:
        dataset_tab.render(data_splits)

    with tab_errors:
        error_tab.render(mlflow_client, data_splits["test"], cfg, device)

    with tab_explain:
        explainability_tab.render(mlflow_client, data_splits, cfg, device)


if __name__ == "__main__":
    main()
