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
