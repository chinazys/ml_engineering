# Lab 01 Report: CIFAR-10 Image Classification Pipeline

## 1. Introduction

This project implements a complete deep learning training pipeline for image classification on the **CIFAR-10** dataset. CIFAR-10 is a well-established benchmark in computer vision containing 60,000 32×32 color images distributed across 10 mutually exclusive classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

**Goals:**
- Build an end-to-end training pipeline: data download → ingestion → augmentation → training → evaluation
- Achieve reasonable classification accuracy with a custom CNN architecture
- Follow software engineering best practices: configuration management, logging, dependency management, code quality tooling, and version control

## 2. Pipeline Description

The pipeline is implemented as a single Jupyter notebook (`ml_engineering_lab_01.ipynb`) with clearly separated stages.

### 2.1 Environment Setup

Poetry is used for dependency management. The notebook generates a `pyproject.toml` at the repository root and runs `poetry install --no-root` to create a reproducible virtual environment. Key dependencies include PyTorch (with CUDA 12.8 support), torchvision, scikit-learn, pandas, numpy, and PyYAML. Development tools (mypy, ruff, black, isort) are installed as dev dependencies.

### 2.2 Configuration Loading

All hyperparameters, data paths, and model settings are stored in `labs/lab01/configs/config.yaml`. The notebook loads this file with `yaml.safe_load()` at startup, ensuring no values are hardcoded. Configuration sections:

- **data**: dataset URL, local directory, validation split ratio, random seed
- **model**: number of output classes
- **training**: batch size, number of workers, epochs, learning rate
- **artifacts**: save directory and model filename

### 2.3 Data Download

The `download_and_extract()` function downloads the CIFAR-10 tar.gz archive from the University of Toronto mirror and extracts it locally. It handles idempotency: if the archive is already downloaded and extracted, it skips redundant work.

### 2.4 Data Ingestion and Splitting

CIFAR-10 stores data in Python pickle files (5 training batches + 1 test batch). The `load_cifar10()` function reads all batches, reshapes images to (N, 3, 32, 32), and normalizes pixel values to [0, 1]. The `train_val_split()` function splits the 50,000 training images into 40,000 train and 10,000 validation samples using a fixed random seed for reproducibility.

Final splits:
- Train: 40,000 images
- Validation: 10,000 images
- Test: 10,000 images

### 2.5 Data Augmentation

Training data is augmented with:
- **RandomHorizontalFlip**: 50% chance of horizontal mirroring
- **RandomCrop(32, padding=4)**: random 32×32 crop from a 40×40 padded image

Both train and evaluation data are normalized using CIFAR-10 channel-wise mean and standard deviation values (mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]).

### 2.6 Model Architecture

`CifarCNN` is a custom CNN with three convolutional blocks:

| Layer | Details |
|-------|---------|
| Conv2d(3→32, 3×3) + BatchNorm + ReLU | First feature extraction block |
| Conv2d(32→64, 3×3) + BatchNorm + ReLU + MaxPool(2) + Dropout2d(0.25) | Second block with downsampling |
| Conv2d(64→128, 3×3) + BatchNorm + ReLU + MaxPool(2) + Dropout2d(0.25) | Third block with downsampling |
| Flatten → Linear(128×8×8→256) → ReLU → Dropout(0.5) → Linear(256→10) | Classifier head |

Design choices:
- **BatchNorm** after each convolution for training stability
- **Dropout2d** (0.25) in feature extractor and **Dropout** (0.5) in classifier to reduce overfitting
- **Adam optimizer** with learning rate 0.001
- **CrossEntropyLoss** as the loss function

### 2.7 Training Loop

The `train_model()` function trains for 20 epochs. Each epoch consists of:
1. **Training phase**: forward pass, loss computation, backpropagation, optimizer step
2. **Validation phase**: forward pass only (no gradients), validation loss computation
3. **Checkpointing**: if validation loss improves, save model weights to `artifacts/best_model.pth`

Logging records train loss and validation loss at every epoch.

### 2.8 Evaluation

After training, the best model checkpoint is loaded and evaluated on the held-out test set. The `test_model()` function computes:
- Test loss
- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score

## 3. Model Evaluation

### 3.1 Results

| Metric    | Value  |
|-----------|--------|
| Test Loss | 0.8000 |
| Accuracy  | 0.7233 |
| Precision | 0.7218 |
| Recall    | 0.7233 |
| F1 Score  | 0.7182 |

### 3.2 Analysis

The model achieves **72.3% accuracy** on the test set. Precision, recall, and F1 scores are closely aligned (~0.72), indicating balanced performance across classes without severe bias toward any particular class.

This accuracy is reasonable for a relatively simple 3-layer CNN trained for 20 epochs. State-of-the-art results on CIFAR-10 exceed 95% using deeper architectures (ResNet, EfficientNet) and more sophisticated training regimes (cosine annealing, mixup, cutout). The gap highlights the trade-off between model simplicity and performance.

### 3.3 Challenges

- **Overfitting**: without Dropout and BatchNorm, the model overfit quickly. Adding regularization improved generalization.
- **Python version compatibility**: `tarfile.extractall(filter="data")` is only available in Python 3.12+, requiring removal for Python 3.11 compatibility.
- **Notebook code quality**: running linters on notebook code required exporting to a script and cleaning cell markers and magic commands.

## 4. Best Practices

### 4.1 Configuration Management

All parameters are stored in `config.yaml` and loaded at runtime. No hyperparameters, paths, or settings are hardcoded in the notebook. This makes experiments reproducible and easy to modify without touching code.

### 4.2 Logging

The Python `logging` module is used throughout the pipeline. No `print()` statements are used for pipeline output. Log messages cover: config loading, data download/extraction status, dataset sizes, epoch-level training/validation losses, model checkpointing, and final test metrics.

### 4.3 Code Quality

Four tools are configured and enforced:
- **isort**: import sorting with black-compatible profile, line length 100
- **ruff**: fast linter; E402 (module-level import not at top) is excluded since notebook exports inherently have non-top-level imports
- **mypy**: static type checking with type hints on all function signatures
- **black**: code formatting with line length 100

All tools pass with zero errors, verified both from the terminal and from within the notebook itself (cell 24).

### 4.4 Dependency Management

Poetry manages all dependencies via `pyproject.toml`. The project pins Python ~3.11 and specifies a supplemental PyTorch index for CUDA 12.8 wheels. A `poetry.lock` file ensures deterministic installs.

### 4.5 Version Control

The project is maintained in a Git repository with meaningful, incremental commits covering:
- Initial skeleton and config
- Full pipeline implementation
- Bug fixes (tarfile, extraction logic, type annotations)
- README updates
- Code quality tooling integration

## 5. Reflection

### What worked well
- **Config-driven design** made it easy to adjust hyperparameters and rerun experiments
- **Logging** provided clear visibility into training progress and debugging
- **Poetry** ensured consistent environments across machines
- **BatchNorm + Dropout** together effectively controlled overfitting

### Potential improvements
- **Deeper architectures**: ResNet-18 or similar would likely push accuracy above 90%
- **Learning rate scheduling**: cosine annealing or step decay could improve convergence
- **Advanced augmentation**: cutout, mixup, or AutoAugment have shown strong results on CIFAR-10
- **More epochs**: 20 epochs is relatively short; 100+ with a scheduler would help
- **Early stopping**: rather than fixed epoch count, stop when validation loss plateaus
- **Experiment tracking**: integrating MLflow or Weights & Biases would provide better metric visualization and comparison
