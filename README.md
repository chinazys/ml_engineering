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
