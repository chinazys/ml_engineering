"""Inference utilities for CIFAR-10 model evaluation."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.data import CifarDataset, normalize_images
from src.model import CifarCNN

logger = logging.getLogger(__name__)


def run_inference(
    model: CifarCNN,
    images: np.ndarray,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model inference on a set of images.

    Args:
        model: Trained CifarCNN in eval mode.
        images: Float32 CHW array in [0, 1] (not yet normalized).
        batch_size: Batch size for inference.
        device: Torch device.

    Returns:
        predictions: Int64 array of predicted class indices.
        probabilities: Float32 array of shape (N, n_classes).
    """
    model.eval()
    normalized = normalize_images(images)
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(normalized), batch_size):
            batch = torch.from_numpy(normalized[start : start + batch_size]).to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    probabilities = np.concatenate(all_probs, axis=0)
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    logger.info("Inference complete: %d samples, device=%s", len(images), device)
    return predictions, probabilities


def find_misclassified(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> List[Dict]:
    """Return list of dicts describing each misclassified sample."""
    mask = true_labels != predictions
    indices = np.where(mask)[0]
    results = []
    for idx in indices:
        top_conf = float(probabilities[idx, predictions[idx]])
        results.append(
            {
                "index": int(idx),
                "true_label": int(true_labels[idx]),
                "pred_label": int(predictions[idx]),
                "confidence": top_conf,
            }
        )
    logger.info("Found %d misclassified samples", len(results))
    return results
