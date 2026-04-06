# Lab 02 Report: Automating Dataset Extension

## 1. Introduction

This project implements an automated dataset extension system for CIFAR-10 image classification. The goal is to demonstrate how configuration-driven batch selection affects model performance by dynamically combining different subsets of training data while keeping the test set static.

**Dataset:** CIFAR-10 — 60,000 32×32 color images across 10 classes. The dataset is pre-split into 5 training batches (10,000 images each) and 1 test batch (10,000 images).

**Goals:**
- Create a configuration-driven system for selecting which data batches compose training and validation sets
- Keep the test set static across all experiments for consistent evaluation
- Train models under 4 different batch configurations and compare performance metrics
- Follow software engineering best practices: configuration management, logging, code quality, dependency management

## 2. Pipeline Description

The pipeline is implemented in a single Jupyter notebook (`ml_engineering_lab_02.ipynb`) with clearly separated stages.

### 2.1 Configuration Management

All parameters are stored in `labs/lab02/configs/config.yaml`. The key addition compared to Lab 01 is the `experiments` section, which defines multiple batch configurations:

```yaml
experiments:
  - name: "1_train_1_val"
    train_batches: [1]
    val_batches: [2]
  - name: "2_train_1_val"
    train_batches: [1, 2]
    val_batches: [3]
  - name: "3_train_1_val"
    train_batches: [1, 2, 3]
    val_batches: [4]
  - name: "4_train_1_val"
    train_batches: [1, 2, 3, 4]
    val_batches: [5]
```

Each experiment specifies which of the 5 CIFAR-10 training batches to use for training and validation. The test batch is always the same static `test_batch` file, ensuring a consistent evaluation baseline.

### 2.2 Data Loading with Batch Selection

The data pipeline consists of three functions:

- **`load_batches(data_dir, batch_ids)`** — loads specific CIFAR-10 batches by their IDs (1–5), concatenates them, reshapes to (N, 3, 32, 32), and normalizes to [0, 1].
- **`load_test_set(data_dir)`** — loads the static test batch (always the same across experiments).
- **`process_experiment_data(data_dir, train_batches, val_batches)`** — orchestrates loading of train, validation, and test sets for a given experiment configuration.

This design replaces Lab 01's `load_cifar10()` + `train_val_split()` approach. Instead of loading all 5 batches and splitting randomly, each experiment selects distinct batches for training and validation, making the data composition fully controlled by configuration.

### 2.3 Model Architecture

The same `CifarCNN` model from Lab 01 is used (3 convolutional blocks with BatchNorm, Dropout, and a 2-layer classifier). This ensures that performance differences across experiments are attributable solely to the training data composition.

### 2.4 Experiment Runner

The `run_experiment()` function handles a single experiment end-to-end:
1. Loads batches per the experiment config
2. Applies data augmentation (RandomHorizontalFlip, RandomCrop) to training data
3. Creates a fresh model, optimizer, and loss function
4. Trains for 20 epochs with validation-loss checkpointing
5. Evaluates the best checkpoint on the static test set

The `main()` orchestrator iterates over all experiments and collects results into a DataFrame.

## 3. Model Evaluation

### 3.1 Results

| Experiment | Train Batches | Train Size | Val Size | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 1_train_1_val | [1] | 10,000 | 10,000 | 0.5816 | 0.5795 | 0.5816 | 0.5767 |
| 2_train_1_val | [1, 2] | 20,000 | 10,000 | 0.5972 | 0.6000 | 0.5972 | 0.5843 |
| 3_train_1_val | [1, 2, 3] | 30,000 | 10,000 | 0.6759 | 0.6719 | 0.6759 | 0.6678 |
| 4_train_1_val | [1, 2, 3, 4] | 40,000 | 10,000 | 0.7158 | 0.7126 | 0.7158 | 0.7110 |

### 3.2 Analysis

The results demonstrate a clear positive correlation between training data size and model performance:

- **1 batch → 4 batches**: accuracy increases from 58.2% to 71.6% (+13.4 percentage points)
- **Diminishing returns**: the jump from 1→2 batches (+1.6%) is much smaller than 2→3 batches (+7.9%), suggesting the 3rd batch brought more diverse or complementary examples
- **3→4 batches** adds another +4.0%, showing continued benefit from more data
- **Precision, recall, and F1** track closely with accuracy (~0.71 across all metrics at 4 batches), indicating balanced class performance without systematic bias

The static test set ensures these comparisons are fair — every experiment is evaluated on exactly the same 10,000 images. The model architecture, optimizer, learning rate, and number of epochs are identical across experiments, isolating the effect of training data volume.

Comparing to Lab 01 (72.3% accuracy with all 5 batches split 80/20), the 4-batch experiment (71.6%) achieves comparable performance, confirming that the batch-selection approach works correctly.

## 4. Best Practices

### 4.1 Configuration Management
All batch selection, training hyperparameters, and artifact paths are controlled via `config.yaml`. Adding a new experiment requires only appending an entry to the `experiments` list — no code changes needed.

### 4.2 Logging
Python's `logging` module is used throughout. Key events logged include: config loading, batch loading (with image counts), data split summaries, per-epoch train/val loss, model checkpointing, and final test metrics.

### 4.3 Code Quality
All notebook code passes:
- **isort** (import sorting, black-compatible profile, line length 100)
- **ruff** (linting, E402 excluded for notebook-style imports)
- **mypy** (static type checking with `--ignore-missing-imports`)

All functions have type annotations.

### 4.4 Dependency Management
Poetry manages all dependencies via `pyproject.toml` at the repo root. The notebook generates this file and runs `poetry install --no-root` to ensure a reproducible environment.

### 4.5 Version Control
Code is committed incrementally to Git with descriptive messages at each milestone.

## 5. Reflection

The configuration-driven batch selection approach is effective and flexible. Key takeaways:

- **Data volume matters**: even with the same model and hyperparameters, going from 10,000 to 40,000 training images improves accuracy by ~13 percentage points
- **Static test set is essential**: without it, comparing experiments would be confounded by different test distributions
- **Config-driven experiments are reproducible**: anyone can replicate the exact experiments by running the notebook with the same config file

**Possible improvements:**
- Add per-class metrics to identify which classes benefit most from additional data
- Test with data augmentation variations per experiment
- Explore batch composition effects (e.g., does batch 3 alone train better than batch 1 alone?)
- Add learning rate scheduling or early stopping for more robust training
