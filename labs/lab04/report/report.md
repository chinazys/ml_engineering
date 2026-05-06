# Lab 04 Report: MLflow Experiment Tracking and Artifact Management

## 1. Introduction

This lab integrates MLflow into the CIFAR-10 training workflow to track experiments, centralize run metadata, and store model artifacts. The objective was to log parameters, epoch-level and final metrics, and output files for multiple runs under one experiment, then verify that a logged model artifact can be downloaded and reused.

## 2. Tracking Server Setup

### Environment and tools

- Python environment: `.venv` (Python 3.11)
- MLflow version: `3.12.0`
- Tracking backend: `sqlite:///mlflow.db`
- Artifact store: `./mlruns`
- Experiment name: `cifar10_lab04`

### Server start command

```bash
.venv/bin/mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
```

### Accessibility verification

- Uvicorn startup log: `Uvicorn running on http://127.0.0.1:5000`
- HTTP check: `curl -I http://127.0.0.1:5000` returned `HTTP/1.1 200 OK`

### Git hygiene

The repository `.gitignore` excludes MLflow runtime files that should not be versioned:

- `mlruns/`
- `mlflow.db`
- `mlflow.db-journal`

## 3. Logging Details

A single config file controls tracking and experiment parameters:

- `labs/lab04/configs/config.yaml`

Logged parameters per run:

- data settings (`data_dir`, `cifar_subdir`, `test_batch`, `train_batches`, `val_batches`)
- model settings (`n_classes`)
- training settings (`batch_size`, `learning_rate`, `num_epochs`, `num_workers`)

Logged metrics:

- Per epoch: `train_loss`, `val_loss`, `val_accuracy`
- Final: `test_loss`, `accuracy`, `precision`, `recall`, `f1`, `best_val_loss`, `best_epoch`

Logged artifacts:

- `model/best_model.pth`
- `config/resolved_config.json`
- `config/config.yaml`

Execution summary file:

- `labs/lab04/results/runs_summary.json`

## 4. Experimentation Process

All runs were logged under one experiment (`cifar10_lab04`) with hierarchical run names:

- `exp_cifar10/run_lr_0.001_bs_128`
- `exp_cifar10/run_lr_0.0005_bs_128`
- `exp_cifar10/run_lr_0.001_bs_64`

Run comparison:

| Run Name | Run ID | Batch Size | Learning Rate | Accuracy | F1 | Test Loss |
|---|---|---:|---:|---:|---:|---:|
| exp_cifar10/run_lr_0.001_bs_128 | 8051b7ed511343b4978c234dce73da85 | 128 | 0.0010 | 0.6199 | 0.6128 | 1.0855 |
| exp_cifar10/run_lr_0.0005_bs_128 | fd87a979795f49e7bba57237413b4246 | 128 | 0.0005 | 0.6366 | 0.6250 | 1.0519 |
| exp_cifar10/run_lr_0.001_bs_64 | 039af0c10ef14b79b77d353a92ebd5fd | 64 | 0.0010 | 0.6186 | 0.6084 | 1.1256 |

Best run by accuracy:

- Run ID: `fd87a979795f49e7bba57237413b4246`
- Accuracy: `0.6366`
- F1: `0.6250`

## 5. Artifact Download and Reuse

To verify artifact usability, the best-run checkpoint was downloaded from MLflow and re-evaluated:

```bash
PYTHONPATH=labs/lab04 .venv/bin/python labs/lab04/scripts/download_and_validate_artifact.py \
  --run-id fd87a979795f49e7bba57237413b4246 \
  --config labs/lab04/configs/config.yaml
```

Validation output (`labs/lab04/results/artifact_validation.json`) confirms consistent metrics:

| Metric | Value |
|---|---:|
| Test Loss | 1.0519 |
| Accuracy | 0.6366 |
| Precision | 0.6302 |
| Recall | 0.6366 |
| F1 | 0.6250 |

The downloaded artifact reproduces the same values as the original run summary for the selected run.

## 6. Reflection

### Benefits

- Centralized experiment history in one UI with searchable run metadata.
- Direct comparison of hyperparameter variants under one experiment.
- Automatic persistence of model checkpoints and run-specific config snapshots.
- Reproducibility supported by run IDs and tracked artifacts.

### Challenges

- MLflow dependency resolution is heavier than previous labs and increases setup time.
- Tracking URI, artifact store, and local paths must be configured consistently to avoid fragmented logs.
- Short training runs (8 epochs) speed up iteration but reduce absolute accuracy compared to longer training.

### Potential improvements

- Add automated hyperparameter sweeps and ranking by a target metric.
- Register the best model in MLflow Model Registry and define a promotion workflow.
- Export MLflow comparisons into CI artifacts for automatic reporting on each push.
