"""MLflow interaction utilities."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import torch

from src.model import CifarCNN

logger = logging.getLogger(__name__)


def set_tracking_uri(uri: str) -> None:
    """Configure the MLflow tracking URI."""
    mlflow.set_tracking_uri(uri)
    logger.info("MLflow tracking URI set to: %s", uri)


def list_experiments(client: mlflow.MlflowClient) -> List[Dict[str, Any]]:
    """Return all experiments as a list of dicts."""
    experiments = client.search_experiments()
    result = [
        {
            "experiment_id": exp.experiment_id,
            "name": exp.name,
            "lifecycle_stage": exp.lifecycle_stage,
        }
        for exp in experiments
        if exp.lifecycle_stage == "active"
    ]
    logger.info("Found %d active experiments", len(result))
    return result


def list_runs(client: mlflow.MlflowClient, experiment_id: str) -> List[Dict[str, Any]]:
    """Return all runs for a given experiment."""
    runs = client.search_runs(experiment_ids=[experiment_id])
    result = [
        {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name or run.info.run_id[:8],
            "status": run.info.status,
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
        }
        for run in runs
    ]
    logger.info("Found %d runs for experiment %s", len(result), experiment_id)
    return result


def get_run_metrics(client: mlflow.MlflowClient, run_id: str) -> Dict[str, float]:
    """Return the final logged metrics for a run."""
    run = client.get_run(run_id)
    return dict(run.data.metrics)


def get_run_params(client: mlflow.MlflowClient, run_id: str) -> Dict[str, str]:
    """Return logged parameters for a run."""
    run = client.get_run(run_id)
    return dict(run.data.params)


def load_model_from_run(
    client: mlflow.MlflowClient,
    run_id: str,
    n_classes: int,
    device: torch.device,
) -> Optional[CifarCNN]:
    """Download the model artifact for a run and load it into CifarCNN."""
    try:
        artifacts = client.list_artifacts(run_id)
        pth_artifacts = [a for a in artifacts if a.path.endswith(".pth")]
        if not pth_artifacts:
            logger.warning("No .pth artifact found for run %s", run_id)
            return None

        artifact_path = pth_artifacts[0].path
        logger.info("Downloading artifact %s from run %s", artifact_path, run_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = client.download_artifacts(run_id, artifact_path, tmp_dir)
            model = CifarCNN(n_classes=n_classes)
            state_dict = torch.load(local_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            logger.info("Model loaded successfully from run %s", run_id)
            return model

    except Exception as exc:
        logger.error("Failed to load model from run %s: %s", run_id, exc)
        return None
