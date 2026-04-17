"""Shared data loading utilities for CIFAR-10 DVC pipeline."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def _unpickle(file_path: str) -> Dict[bytes, Any]:
    """Load a CIFAR-10 batch file."""
    with open(file_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    return batch  # type: ignore[no-any-return]


def load_batches(
    data_dir: str, cifar_subdir: str, batch_ids: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load specific CIFAR-10 training batches by their IDs (1-5).

    Args:
        data_dir: Path to the data directory.
        cifar_subdir: Name of the CIFAR-10 subdirectory.
        batch_ids: List of batch numbers to load (e.g. [1, 2, 3]).

    Returns:
        Tuple of (images, labels).
        Images shape: (N, 3, 32, 32), dtype float32, normalized to [0, 1].
    """
    cifar_dir = Path(data_dir) / cifar_subdir
    images_list: List[np.ndarray] = []
    labels_list: List[int] = []

    for batch_id in batch_ids:
        batch_file = cifar_dir / f"data_batch_{batch_id}"
        if not batch_file.exists():
            raise FileNotFoundError(f"Batch file not found: {batch_file}")
        batch = _unpickle(str(batch_file))
        images_list.append(batch[b"data"])
        labels_list.extend(batch[b"labels"])

    images = np.concatenate(images_list).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.array(labels_list, dtype=np.int64)

    logger.info(f"Loaded batches {batch_ids}: {images.shape[0]} images")
    return images, labels


def load_test_set(
    data_dir: str, cifar_subdir: str, test_batch_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the static CIFAR-10 test batch."""
    cifar_dir = Path(data_dir) / cifar_subdir
    test_batch = _unpickle(str(cifar_dir / test_batch_name))
    test_images = test_batch[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    test_labels = np.array(test_batch[b"labels"], dtype=np.int64)

    logger.info(f"Loaded test set: {test_images.shape[0]} images")
    return test_images, test_labels


class CifarDataset(Dataset):  # type: ignore[type-arg]
    """Dataset for CIFAR-10 numpy arrays."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Optional[nn.Module] = None,
    ) -> None:
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = int(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label


def create_data_loader(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    num_workers: int,
    transform: Optional[nn.Module] = None,
    shuffle: bool = True,
) -> DataLoader:  # type: ignore[type-arg]
    """Create a DataLoader from numpy arrays."""
    dataset = CifarDataset(images, labels, transform=transform)
    data_loader: DataLoader = DataLoader(  # type: ignore[type-arg]
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return data_loader
