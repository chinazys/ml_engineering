"""DVC Stage 3: Evaluate trained CIFAR-10 CNN model on the static test set."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data import create_data_loader, load_test_set
from src.model import CifarCNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model(
    model: nn.Module,
    test_loader: DataLoader,  # type: ignore[type-arg]
    loss_function: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on test set. Returns loss, accuracy, precision, recall, F1."""
    model.eval()
    test_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            test_loss += loss_function(outputs, targets).item()

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    test_loss /= len(test_loader)
    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)

    metrics: Dict[str, float] = {
        "test_loss": round(test_loss, 4),
        "accuracy": round(float(accuracy_score(all_targets_arr, all_preds_arr)), 4),
        "precision": round(
            float(precision_score(all_targets_arr, all_preds_arr, average="weighted")), 4
        ),
        "recall": round(float(recall_score(all_targets_arr, all_preds_arr, average="weighted")), 4),
        "f1": round(float(f1_score(all_targets_arr, all_preds_arr, average="weighted")), 4),
    }

    for name, value in metrics.items():
        logger.info(f"{name}: {value:.4f}")

    return metrics


def main() -> None:
    """Evaluate the trained model and save metrics to JSON."""
    logger.info("Starting evaluation stage...")

    with open("params.yaml") as f:
        params: Dict[str, Any] = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load test data
    data_cfg = params["data"]
    test_images, test_labels = load_test_set(
        data_cfg["data_dir"], data_cfg["cifar_subdir"], data_cfg["test_batch"]
    )

    eval_transform = transforms.Compose([
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])

    train_cfg = params["training"]
    test_loader = create_data_loader(
        test_images, test_labels,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        transform=eval_transform,
        shuffle=False,
    )

    # Load trained model
    model_path = Path(params["artifacts"]["save_dir"]) / params["artifacts"]["best_model_filename"]
    model = CifarCNN(n_classes=params["model"]["n_classes"])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    logger.info(f"Model loaded from {model_path}")

    # Evaluate
    loss_function = nn.CrossEntropyLoss()
    metrics = test_model(model, test_loader, loss_function, device)

    # Save metrics
    metrics_path = Path(params["artifacts"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    logger.info("Evaluation stage complete.")


if __name__ == "__main__":
    main()
