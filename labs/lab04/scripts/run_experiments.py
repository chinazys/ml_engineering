"""Run CIFAR-10 experiments with MLflow tracking."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.data import create_data_loader, load_batches, load_test_set
from src.model import CifarCNN
from torch.utils.data import DataLoader
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_one_run(
    model: nn.Module,
    train_loader: DataLoader,  # type: ignore[type-arg]
    val_loader: DataLoader,  # type: ignore[type-arg]
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
) -> Tuple[float, int]:
    """Train one run and return best validation loss and epoch index."""
    model.to(device)
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            outputs = model(batch_inputs)
            loss = loss_function(outputs, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_preds: List[int] = []
        val_targets: List[int] = []

        with torch.no_grad():
            for val_inputs, val_targets_batch in val_loader:
                val_inputs = val_inputs.to(device)
                val_targets_batch = val_targets_batch.to(device)
                val_outputs = model(val_inputs)

                val_loss += loss_function(val_outputs, val_targets_batch).item()
                val_preds.extend(val_outputs.argmax(dim=1).cpu().numpy())
                val_targets.extend(val_targets_batch.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = float(accuracy_score(np.array(val_targets), np.array(val_preds)))

        mlflow.log_metric("train_loss", avg_train_loss, step=epoch + 1)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch + 1)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch + 1)

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} | train_loss={avg_train_loss:.4f} "
            f"val_loss={avg_val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1

    return best_val_loss, best_epoch


def evaluate_on_test(
    model: nn.Module,
    test_loader: DataLoader,  # type: ignore[type-arg]
    loss_function: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on test set and return aggregated metrics."""
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

    metrics: Dict[str, float] = {
        "test_loss": round(float(test_loss), 4),
        "accuracy": round(float(accuracy_score(targets, preds)), 4),
        "precision": round(float(precision_score(targets, preds, average="weighted")), 4),
        "recall": round(float(recall_score(targets, preds, average="weighted")), 4),
        "f1": round(float(f1_score(targets, preds, average="weighted")), 4),
    }
    return metrics


def sanitize_run_name(run_name: str) -> str:
    """Make run name safe for filesystem paths."""
    return run_name.replace(" ", "_").replace("/", "_")


def main() -> None:
    """Execute all configured experiments and log them to MLflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="labs/lab04/configs/config.yaml",
        help="Path to Lab 04 config file",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    mlflow_cfg = cfg["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])

    experiment_name = mlflow_cfg["experiment_name"]
    artifact_location = mlflow_cfg.get("artifact_location")
    existing_experiment = mlflow.get_experiment_by_name(experiment_name)
    if existing_experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)

    logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
    logger.info(f"Experiment: {experiment_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_cfg = cfg["data"]
    common_training_cfg = cfg["training"]
    model_cfg = cfg["model"]

    test_images, test_labels = load_test_set(
        data_cfg["data_dir"], data_cfg["cifar_subdir"], data_cfg["test_batch"]
    )

    eval_transform = transforms.Compose([
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])

    run_summaries: List[Dict[str, Any]] = []

    for run_cfg in cfg["experiments"]:
        run_name = f"exp_cifar10/{run_cfg['name']}"
        logger.info(f"Starting run: {run_name}")

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "data_dir": data_cfg["data_dir"],
                    "cifar_subdir": data_cfg["cifar_subdir"],
                    "test_batch": data_cfg["test_batch"],
                    "train_batches": str(run_cfg["train_batches"]),
                    "val_batches": str(run_cfg["val_batches"]),
                    "n_classes": model_cfg["n_classes"],
                    "batch_size": run_cfg["batch_size"],
                    "learning_rate": run_cfg["learning_rate"],
                    "num_epochs": common_training_cfg["num_epochs"],
                    "num_workers": common_training_cfg["num_workers"],
                }
            )

            train_images, train_labels = load_batches(
                data_cfg["data_dir"], data_cfg["cifar_subdir"], run_cfg["train_batches"]
            )
            val_images, val_labels = load_batches(
                data_cfg["data_dir"], data_cfg["cifar_subdir"], run_cfg["val_batches"]
            )

            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
            ])

            train_loader = create_data_loader(
                train_images,
                train_labels,
                batch_size=run_cfg["batch_size"],
                num_workers=common_training_cfg["num_workers"],
                transform=train_transform,
                shuffle=True,
            )
            val_loader = create_data_loader(
                val_images,
                val_labels,
                batch_size=run_cfg["batch_size"],
                num_workers=common_training_cfg["num_workers"],
                transform=eval_transform,
                shuffle=False,
            )
            test_loader = create_data_loader(
                test_images,
                test_labels,
                batch_size=run_cfg["batch_size"],
                num_workers=common_training_cfg["num_workers"],
                transform=eval_transform,
                shuffle=False,
            )

            model = CifarCNN(n_classes=model_cfg["n_classes"])
            loss_function = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=run_cfg["learning_rate"])

            best_val_loss, best_epoch = train_one_run(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_function=loss_function,
                optimizer=optimizer,
                num_epochs=common_training_cfg["num_epochs"],
                device=device,
            )

            metrics = evaluate_on_test(
                model=model,
                test_loader=test_loader,
                loss_function=loss_function,
                device=device,
            )
            metrics["best_val_loss"] = round(float(best_val_loss), 4)
            metrics["best_epoch"] = float(best_epoch)

            mlflow.log_metrics(metrics)

            run = mlflow.active_run()
            if run is None:
                raise RuntimeError("No active MLflow run found")

            run_dir = Path("artifacts/lab04") / sanitize_run_name(run_name)
            run_dir.mkdir(parents=True, exist_ok=True)

            model_path = run_dir / "best_model.pth"
            torch.save(model.state_dict(), model_path)

            resolved_config_path = run_dir / "resolved_config.json"
            with open(resolved_config_path, "w") as f:
                json.dump(
                    {
                        "mlflow": mlflow_cfg,
                        "data": data_cfg,
                        "model": model_cfg,
                        "training": common_training_cfg,
                        "run": run_cfg,
                    },
                    f,
                    indent=2,
                )

            mlflow.log_artifact(str(model_path), artifact_path="model")
            mlflow.log_artifact(str(resolved_config_path), artifact_path="config")
            mlflow.log_artifact(args.config, artifact_path="config")

            run_summary: Dict[str, Any] = {
                "run_id": run.info.run_id,
                "run_name": run_name,
                "params": {
                    "batch_size": run_cfg["batch_size"],
                    "learning_rate": run_cfg["learning_rate"],
                    "num_epochs": common_training_cfg["num_epochs"],
                    "train_batches": run_cfg["train_batches"],
                    "val_batches": run_cfg["val_batches"],
                },
                "metrics": metrics,
            }
            run_summaries.append(run_summary)

            logger.info(
                f"Completed run {run_name} | run_id={run.info.run_id} "
                f"test_acc={metrics['accuracy']:.4f}"
            )

    results_dir = Path("labs/lab04/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "runs_summary.json"
    with open(summary_path, "w") as f:
        json.dump(run_summaries, f, indent=2)

    logger.info(f"Saved run summary to {summary_path}")


if __name__ == "__main__":
    main()
