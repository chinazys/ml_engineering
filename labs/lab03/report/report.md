# Lab 03 Report: Data Version Control (DVC) Pipeline Automation

## 1. Introduction

The CIFAR-10 training pipeline from Labs 01–02 was refactored from a Jupyter notebook into three Python scripts orchestrated by DVC. Large files (dataset, model checkpoint) are tracked by DVC; code, `dvc.yaml`, `dvc.lock`, `params.yaml`, and `metrics.json` are tracked by Git. Running `dvc repro` re-executes only stages whose inputs changed.

## 2. Pipeline Description

The DVC pipeline is defined in `dvc.yaml` at the repository root and consists of three stages executed sequentially via dependency resolution.

### 2.1 Stage: `download_data`

- **Script:** `labs/lab03/scripts/download_data.py`
- **Parameters:** `data.data_dir`, `data.dataset_url`, `data.cifar_subdir`
- **Outputs:** `data/cifar-10-batches-py/` (DVC-tracked)

Downloads the CIFAR-10 archive from the configured URL, extracts it, and deletes the archive. The stage is idempotent: if the output directory already exists, DVC skips re-execution entirely.

### 2.2 Stage: `train`

- **Script:** `labs/lab03/scripts/train.py`
- **Dependencies:** `data/cifar-10-batches-py/`, `labs/lab03/src/data.py`, `labs/lab03/src/model.py`
- **Parameters:** `data.*`, `model.*`, `training.*`, `artifacts.*`
- **Outputs:** `artifacts/lab03/best_model.pth` (DVC-tracked)

Loads CIFAR-10 batches specified by `data.train_batches` and `data.val_batches`, applies augmentation (RandomHorizontalFlip, RandomCrop with padding=4, channel-wise normalization), trains `CifarCNN` for `training.num_epochs` epochs using Adam optimizer, and saves the checkpoint with the lowest validation loss.

### 2.3 Stage: `evaluate`

- **Script:** `labs/lab03/scripts/evaluate.py`
- **Dependencies:** `artifacts/lab03/best_model.pth`, `data/cifar-10-batches-py/`, source modules
- **Parameters:** `data.*`, `model.*`, `training.*`, `artifacts.*`
- **Metrics:** `labs/lab03/metrics.json` (Git-tracked, `cache: false`)

Loads the best checkpoint determined by the `train` stage, runs inference on the static test set (`test_batch`), and writes five metrics to `metrics.json`: test loss, accuracy, weighted precision, weighted recall, and weighted F1-score. Since `cache: false` is set, DVC does not store this file in the cache — it is committed directly to Git and viewable via `dvc metrics show`.

### 2.4 Dependency graph and selective re-execution

```
download_data  →  train  →  evaluate
```

`dvc repro` resolves the dependency graph and re-executes only invalidated stages:

| Change | Stages re-executed |
|---|---|
| `data.dataset_url` modified | all three |
| `training.learning_rate` modified | `train` → `evaluate` |
| `evaluate.py` source modified | `evaluate` only |
| No changes | none (pipeline is up to date) |

State is recorded in `dvc.lock`, which stores the MD5/SHA-256 hashes of every dependency and output. This file is committed to Git, making each pipeline run fully reproducible at any point in the commit history.

## 3. Parameterization

All configurable values are centralized in `params.yaml` at the repository root:

```yaml
data:
  data_dir: data
  dataset_url: "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
  cifar_subdir: cifar-10-batches-py
  test_batch: test_batch
  train_batches: [1, 2, 3, 4]
  val_batches: [5]

model:
  n_classes: 10

training:
  batch_size: 128
  num_workers: 4
  num_epochs: 20
  learning_rate: 0.001

artifacts:
  save_dir: artifacts/lab03
  best_model_filename: best_model.pth
  metrics_path: labs/lab03/metrics.json
```

Each DVC stage declares which parameter groups it consumes via the `params` field in `dvc.yaml`. DVC hashes these parameter values and stores them in `dvc.lock`. When a parameter changes, only the stages that depend on it are invalidated.

**Parameter–stage dependency matrix:**

| Parameter group | `download_data` | `train` | `evaluate` |
|---|:---:|:---:|:---:|
| `data` | ✓ | ✓ | ✓ |
| `model` | | ✓ | ✓ |
| `training` | | ✓ | ✓ |
| `artifacts` | | ✓ | ✓ |

No values are hardcoded in any script. Scripts read `params.yaml` via `yaml.safe_load()` at startup and derive all paths, hyperparameters, and batch selections from it.

### Evaluation results

The pipeline produces the following metrics (stored in `labs/lab03/metrics.json`):

| Metric    | Value  |
|-----------|--------|
| Test Loss | 0.8377 |
| Accuracy  | 0.7127 |
| Precision | 0.7117 |
| Recall    | 0.7127 |
| F1 Score  | 0.7075 |

These values are consistent with Lab 01 (72.3% accuracy, all 5 batches, random 80/20 split) and Lab 02's 4-batch experiment (71.6% accuracy), confirming that the refactoring into DVC-managed scripts preserved model behavior.

## 4. Reflection

### Benefits

- **Reproducibility.** `dvc.lock` records SHA-256 hashes of every dependency and output. Checking out any past commit and running `dvc repro` reconstructs the exact pipeline state — data, model, and metrics.
- **Selective re-execution.** DVC's dependency graph avoids redundant computation. Changing a training hyperparameter re-runs `train` and `evaluate` but skips the ~170 MB data download.
- **Separation of large files from code.** Git remains lightweight; datasets and model checkpoints are stored in the DVC cache (`.dvc/cache/`) and can be pushed to remote storage for collaboration.
- **Parameter tracking.** `dvc metrics show` and `dvc params diff` provide immediate comparison between pipeline runs without manual bookkeeping.

### Challenges

- **Subprocess environment.** DVC executes stage commands as subprocesses using the system Python, not the active Poetry virtualenv. This required specifying `.venv/bin/python` explicitly in `dvc.yaml` commands.
- **PYTHONPATH for local modules.** Source modules under `labs/lab03/src/` are not on the default Python path. The `PYTHONPATH=labs/lab03` environment variable had to be prepended to each stage command.
- **Initial pipeline debugging.** Iterating on `dvc.yaml` stage definitions (correct `deps`, `outs`, `params` declarations) required several cycles of `dvc repro` to identify missing or incorrect dependencies.

### Possible improvements

- Configure a remote DVC storage backend (S3, GCS, or SSH) to enable multi-machine collaboration and CI/CD integration.
- Use `dvc exp run` with parameter overrides to systematically sweep hyperparameters (e.g., learning rate, batch size) without manual `params.yaml` edits.
- Add a data validation stage between download and training to detect schema drift or distribution shifts.
- Integrate `dvc repro` into a GitHub Actions workflow to enforce pipeline reproducibility on every push.
