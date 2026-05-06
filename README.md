# ML Engineering Labs

## Lab 1: CIFAR-10 Image Classification Pipeline

A complete deep learning training pipeline for CIFAR-10 image classification using PyTorch.

### Dataset
- **CIFAR-10**: 60,000 32×32 color images across 10 classes
- Split: 40,000 train / 10,000 validation / 10,000 test

### Results
| Metric    | Value  |
|-----------|--------|
| Test Loss | 0.8000 |
| Accuracy  | 0.7233 |
| Precision | 0.7218 |
| Recall    | 0.7233 |
| F1 Score  | 0.7182 |

### Repository Structure
```
├── pyproject.toml                        # Poetry dependencies & tool config
├── poetry.lock
├── docs/requirements/                    # Assignment PDFs
├── labs/lab01/
│   ├── configs/config.yaml               # Pipeline configuration (YAML)
│   ├── notebooks/ml_engineering_lab_01.ipynb  # Main training notebook
│   └── report/                           # Lab report
├── labs/lab02/
│   ├── configs/config.yaml               # Batch selection & experiment config
│   ├── notebooks/ml_engineering_lab_02.ipynb  # Dataset extension notebook
│   └── report/                           # Lab report
├── labs/lab03/
│   ├── scripts/                          # DVC pipeline stage scripts
│   ├── src/                              # Shared modules (data, model)
│   └── report/                           # Lab report
├── labs/lab04/
│   ├── configs/                          # MLflow experiment configuration
│   ├── scripts/                          # Experiment runner and artifact validation
│   ├── src/                              # Shared modules (data, model)
│   ├── results/                          # Run summaries and artifact validation output
│   └── report/                           # Lab report
├── labs/lab05/
│   ├── configs/                          # W&B experiment configuration
│   ├── scripts/                          # Experiment runner
│   ├── src/                              # Shared modules (data, model)
│   ├── results/                          # Run summaries
│   └── report/                           # Lab report
├── dvc.yaml                              # DVC pipeline definition
├── params.yaml                           # Centralized parameters
```

### Setup
```bash
# Requires Python ~3.11 and Poetry
poetry install --no-root
```

### Key Features
- Config-driven pipeline (YAML) — no hardcoded values
- Python `logging` module throughout (no `print`)
- Data augmentation (random horizontal flip, random crop)
- CNN model with BatchNorm and Dropout
- Best model checkpoint saving
- Full evaluation metrics: accuracy, precision, recall, F1
- Poetry for dependency management
- Code quality tools: mypy, ruff, black, isort

### Code Quality

Notebook code was exported and verified with all configured linters/formatters:

```
$ isort --check --profile black --line-length 100 nb_check.py
✓ No issues found

$ ruff check --ignore E402 nb_check.py
All checks passed!

$ mypy --ignore-missing-imports nb_check.py
Success: no issues found in 1 source file
```

- **isort**: imports sorted (stdlib → third-party → local), black-compatible profile
- **ruff**: all rules pass (E402 excluded — expected in notebook-exported scripts)
- **mypy**: full type-check pass with type hints on all functions
- **black**: line length 100; only cosmetic blank-line diffs from notebook cell boundaries

---

## Lab 2: Automating Dataset Extension

Configuration-driven batch selection to study how training data volume affects model performance.

### Approach
- Select which CIFAR-10 batches (1–5) go to training vs validation via `config.yaml`
- Test set is always the static `test_batch` (10,000 images)
- Run 4 experiments with increasing training data (1→4 batches)

### Results
| Experiment | Train Size | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 1_train_1_val | 10,000 | 0.5816 | 0.5795 | 0.5816 | 0.5767 |
| 2_train_1_val | 20,000 | 0.5972 | 0.6000 | 0.5972 | 0.5843 |
| 3_train_1_val | 30,000 | 0.6759 | 0.6719 | 0.6759 | 0.6678 |
| 4_train_1_val | 40,000 | 0.7158 | 0.7126 | 0.7158 | 0.7110 |

### Key Features
- Config-driven experiment definitions — add experiments without code changes
- Dynamic train/val sets with static test set for fair comparison
- Automated multi-experiment runner with results aggregation
- Performance comparison charts (matplotlib)
- Same code quality standards as Lab 1 (isort, ruff, mypy all pass)

---

## Lab 3: DVC Pipeline Automation

Refactored the training pipeline into three DVC-managed stages: `download_data` → `train` → `evaluate`. All parameters in `params.yaml`; `dvc repro` re-runs only invalidated stages.

### Results
| Metric    | Value  |
|-----------|--------|
| Test Loss | 0.8377 |
| Accuracy  | 0.7127 |
| Precision | 0.7117 |
| Recall    | 0.7127 |
| F1 Score  | 0.7075 |

### Key Features
- 3-stage DVC pipeline with explicit dependency tracking
- `params.yaml` for centralized hyperparameter management
- `dvc.lock` for full reproducibility
- Metrics tracked via `dvc metrics show`
- Code quality: isort, ruff, mypy all pass on all source files

---

## Lab 4: MLflow Experiment Tracking

Integrated MLflow into CIFAR-10 training with a local tracking server, multi-run experiment logging, and model artifact validation.

### Results (3 runs)
| Run Name | Learning Rate | Batch Size | Accuracy | F1 | Test Loss |
|---|---:|---:|---:|---:|---:|
| exp_cifar10/run_lr_0.001_bs_128 | 0.0010 | 128 | 0.6199 | 0.6128 | 1.0855 |
| exp_cifar10/run_lr_0.0005_bs_128 | 0.0005 | 128 | 0.6366 | 0.6250 | 1.0519 |
| exp_cifar10/run_lr_0.001_bs_64 | 0.0010 | 64 | 0.6186 | 0.6084 | 1.1256 |

Best run: `fd87a979795f49e7bba57237413b4246` (`exp_cifar10/run_lr_0.0005_bs_128`)

### Key Features
- MLflow tracking server with SQLite backend and local artifact store
- Per-epoch metric logging (`train_loss`, `val_loss`, `val_accuracy`)
- Final test metrics logging (`accuracy`, `precision`, `recall`, `f1`, `test_loss`)
- Model artifact logging and download-based validation
- Hierarchical run names for easier comparison in MLflow UI

---

## Lab 5: Weights & Biases Experiment Tracking

Integrated Weights & Biases into CIFAR-10 training with local offline logging, structured run naming, per-epoch metrics, and model artifact management.

### Results (3 runs)
| Run Name | Learning Rate | Batch Size | Accuracy | F1 | Test Loss |
|---|---:|---:|---:|---:|---:|
| Run 1 - Default (lr=0.001, bs=128) | 0.0010 | 128 | 0.5930 | 0.5823 | 1.1332 |
| Run 2 - Lower LR (lr=0.0005, bs=128) | 0.0005 | 128 | 0.6585 | 0.6510 | 0.9622 |
| Run 3 - Smaller Batch (lr=0.001, bs=64) | 0.0010 | 64 | 0.6270 | 0.6144 | 1.0410 |

Best run: **Run 2 - Lower LR (lr=0.0005, bs=128)**

### Key Features
- W&B offline-first logging to `./wandb/` directory
- Per-epoch metrics tracked via `wandb.log()` with step counter
- Final test metrics and best validation loss logged per run
- Model artifacts stored locally (`artifacts/lab05/<run_name>/best_model.pth`)
- Structured run names encode hyperparameter decisions
- Ready for cloud synchronization via `wandb sync` with API key
