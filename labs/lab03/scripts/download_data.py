"""DVC Stage 1: Download and extract the CIFAR-10 dataset."""

import logging
import tarfile
from pathlib import Path

import requests
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_and_extract(url: str, save_dir: str) -> str:
    """Download a file from URL and extract if it is a tar archive."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    filename = url.split("/")[-1]
    file_path = save_path / filename

    if not file_path.exists():
        logger.info(f"Downloading '{filename}' from '{url}'...")
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)
        logger.info(f"Download complete: {file_path}")

    if file_path.exists() and filename.endswith((".tar.gz", ".tgz")):
        logger.info(f"Extracting '{filename}'...")
        with tarfile.open(file_path, "r:gz") as tar_ref:
            tar_ref.extractall(save_path)
        file_path.unlink()
        logger.info("Extraction complete.")

    return str(save_path)


def main() -> None:
    """Download CIFAR-10 data based on params.yaml configuration."""
    logger.info("Starting data download stage...")

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    data_dir = params["data"]["data_dir"]
    dataset_url = params["data"]["dataset_url"]
    cifar_subdir = params["data"]["cifar_subdir"]

    cifar_path = Path(data_dir) / cifar_subdir
    if cifar_path.exists():
        logger.info(f"Data already exists at '{cifar_path}'. Skipping download.")
    else:
        download_and_extract(dataset_url, data_dir)

    logger.info("Data download stage complete.")


if __name__ == "__main__":
    main()
