"""Download an MLflow model artifact and validate it on CIFAR-10 test set."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.data import create_data_loader, load_test_set
from src.model import CifarCNN
from torch.utils.data import DataLoader
from torchvision import transforms


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,  # type: ignore[type-arg]
    loss_function: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate downloaded model on static test set."""
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
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_loss /= len(test_loader)
    preds = np.array(all_preds)
    targets = np.array(all_targets)

    return {
        "test_loss": round(float(test_loss), 4),
        "accuracy": round(float(accuracy_score(targets, preds)), 4),
        "precision": round(float(precision_score(targets, preds, average="weighted")), 4),
        "recall": round(float(recall_score(targets, preds, average="weighted")), 4),
        "f1": round(float(f1_score(targets, preds, average="weighted")), 4),
    }


def main() -> None:
    """Download model artifact by run_id and validate it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--config", type=str, default="labs/lab04/configs/config.yaml")
    parser.add_argument(
        "--output",
        type=str,
        default="labs/lab04/results/artifact_validation.json",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])

    local_artifact_path = mlflow.artifacts.download_artifacts(
        run_id=args.run_id,
        artifact_path="model/best_model.pth",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    test_images, test_labels = load_test_set(
        data_cfg["data_dir"], data_cfg["cifar_subdir"], data_cfg["test_batch"]
    )
    eval_transform = transforms.Compose([
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])
    test_loader = create_data_loader(
        test_images,
        test_labels,
        batch_size=128,
        num_workers=train_cfg["num_workers"],
        transform=eval_transform,
        shuffle=False,
    )

    model = CifarCNN(n_classes=cfg["model"]["n_classes"])
    model.load_state_dict(torch.load(local_artifact_path, weights_only=True))
    model.to(device)

    metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    result = {
        "run_id": args.run_id,
        "downloaded_artifact_path": local_artifact_path,
        "metrics": metrics,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
