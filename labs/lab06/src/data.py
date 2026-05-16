"""Data loading utilities for CIFAR-10."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _unpickle(file_path: str) -> Dict[bytes, Any]:
    """Load a CIFAR-10 batch file."""
    with open(file_path, "rb") as f:
        return pickle.load(f, encoding="bytes")  # type: ignore[no-any-return]


def load_batches(
    data_dir: str, cifar_subdir: str, batch_ids: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Load specific CIFAR-10 training batches by IDs (1-5)."""
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
    logger.info("Loaded batches %s: %d images", batch_ids, images.shape[0])
    return images, labels


def load_test_set(
    data_dir: str, cifar_subdir: str, test_batch_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the static CIFAR-10 test batch."""
    cifar_dir = Path(data_dir) / cifar_subdir
    batch = _unpickle(str(cifar_dir / test_batch_name))
    images = batch[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.array(batch[b"labels"], dtype=np.int64)
    logger.info("Loaded test set: %d images", images.shape[0])
    return images, labels


class CifarDataset(Dataset):  # type: ignore[type-arg]
    """PyTorch dataset wrapping CIFAR-10 numpy arrays."""

    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.images[idx], int(self.labels[idx])


def create_data_loader(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 256,
    num_workers: int = 0,
    shuffle: bool = False,
) -> DataLoader:  # type: ignore[type-arg]
    """Create a DataLoader from numpy image/label arrays."""
    dataset = CifarDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Apply CIFAR-10 channel-wise normalization in-place (returns new array)."""
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)
    return (images - mean) / std


def to_display_hwc(image_chw: np.ndarray) -> np.ndarray:
    """Convert a CHW float32 image in [0,1] to HWC uint8 for display."""
    img = np.transpose(image_chw, (1, 2, 0))
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img
