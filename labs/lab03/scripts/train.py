"""DVC Stage 2: Train CIFAR-10 CNN model."""

import logging
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data import create_data_loader, load_batches
from src.model import CifarCNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,  # type: ignore[type-arg]
    val_loader: DataLoader,  # type: ignore[type-arg]
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_path: Path,
) -> Path:
    """Train the model and save the best checkpoint based on validation loss."""
    model.to(device)
    best_val_loss: float = float("inf")

    for epoch in range(num_epochs):
        # Training phase
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
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)
                val_outputs = model(val_inputs)
                val_loss += loss_function(val_outputs, val_targets).item()

        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"Best model saved (val_loss={best_val_loss:.4f})")

    logger.info("Training complete.")
    return save_path


def main() -> None:
    """Train CIFAR-10 model based on params.yaml configuration."""
    logger.info("Starting training stage...")

    with open("params.yaml") as f:
        params: Dict[str, Any] = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    data_cfg = params["data"]
    train_images, train_labels = load_batches(
        data_cfg["data_dir"], data_cfg["cifar_subdir"], data_cfg["train_batches"]
    )
    val_images, val_labels = load_batches(
        data_cfg["data_dir"], data_cfg["cifar_subdir"], data_cfg["val_batches"]
    )
    logger.info(f"Data loaded — train: {len(train_labels)}, val: {len(val_labels)}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])
    eval_transform = transforms.Compose([
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])

    # DataLoaders
    train_cfg = params["training"]
    train_loader = create_data_loader(
        train_images, train_labels,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        transform=train_transform,
        shuffle=True,
    )
    val_loader = create_data_loader(
        val_images, val_labels,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        transform=eval_transform,
        shuffle=False,
    )

    # Model, loss, optimizer
    model = CifarCNN(n_classes=params["model"]["n_classes"])
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    # Train
    artifacts_dir = Path(params["artifacts"]["save_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    save_path = artifacts_dir / params["artifacts"]["best_model_filename"]

    train_model(
        model, train_loader, val_loader, loss_function, optimizer,
        num_epochs=train_cfg["num_epochs"], device=device, save_path=save_path,
    )

    logger.info("Training stage complete.")


if __name__ == "__main__":
    main()
